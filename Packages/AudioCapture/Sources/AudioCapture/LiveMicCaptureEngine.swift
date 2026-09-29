import AudioToolbox
@preconcurrency import AVFoundation
import CoreAudio
import Foundation
import os
import Synchronization

private let logger = Logger(subsystem: "net.scosman.biscotti.audiocapture", category: "LiveMicCapture")

/// Live microphone capture via **VoiceProcessingIO** (`AVAudioEngine`
/// with `inputNode.setVoiceProcessingEnabled(true)`).
///
/// VPIO is the only route to loud, normalised, noise-suppressed mono.
/// Thin hardware adapter -- orchestration lives in `AudioRecorder`.
final class LiveMicCaptureEngine: CaptureEngine, @unchecked Sendable { // swiftlint:disable:this type_body_length
    private let encoder: EncoderSettings
    private let processingFormat: AVAudioFormat

    /// Atomic file ref (bit-pattern of the opaque pointer, 0 = nil).
    /// Lock-free so the real-time tap can read without blocking.
    private let atomicFileRef = Atomic<UInt>(0)

    /// Serializes the tap's `ExtAudioFileWrite` against `closeExtFile()`'s
    /// `ExtAudioFileDispose`. The tap takes it with `trylock` (never blocks on
    /// the real-time thread); `closeExtFile` takes it with `lock` so dispose
    /// waits for any in-flight write. Without this barrier, `engine.stop()` /
    /// `removeTap` is not guaranteed to drain an in-flight render callback, so
    /// dispose could free the AAC encoder while the audio thread is mid-write
    /// — a heap use-after-free.
    private var _fileLock = os_unfair_lock()

    /// Atomic capturing flag -- safe from async contexts.
    private let capturingFlag = Atomic<Bool>(false)

    /// `.userInitiated` matches the QoS of the `AudioRecorder` actor tasks that
    /// `await` start()/stop()/reconnect(): without an explicit QoS the queue runs
    /// at Default, so a user-initiated task awaiting it is a priority inversion
    /// (the runtime flags "User-initiated … waiting on a … Default QoS thread").
    private let engineQueue = DispatchQueue(
        label: "net.scosman.biscotti.mic.engine", qos: .userInitiated
    )
    private var isTearingDown = false
    private var engine: AVAudioEngine?
    private var silenceNode: AVAudioSourceNode?
    private var configObserver: NSObjectProtocol?
    private var outputRateOverride: (deviceID: AudioObjectID, originalRate: Double)?
    private var cachedConverter: AVAudioConverter?
    private var cachedConverterSourceHash: Int = 0

    var onUnrecoverableError: (@Sendable (Error) -> Void)?

    /// Callback fired once after the first audio buffer is written successfully.
    /// Argument: host-clock anchor (seconds) — the recording's t=0.
    /// Route-change rebuilds do NOT re-fire this.
    ///
    /// **Intentional unsynchronised access:** this `var` is written by
    /// `setOnFirstBuffer` (from the `AudioRecorder` actor) and read on the
    /// real-time audio thread in `notifyFirstBufferIfNeeded`. A lock is NOT
    /// used because taking one on the audio thread risks priority inversion
    /// and glitches. The race is benign: Apple-silicon pointer-sized loads
    /// are atomic (no torn read), `didNotifyFirstBuffer` prevents double-fire,
    /// and optional chaining handles the nil case. This mirrors AudioLab's
    /// validated `VPIOMicCapture.onStarted` pattern. Do NOT "fix" with a lock.
    private var onFirstBuffer: (@Sendable (Double) -> Void)?

    func setOnFirstBuffer(_ callback: (@Sendable (Double) -> Void)?) {
        onFirstBuffer = callback
    }

    /// Guards one-shot firing of `onFirstBuffer`. Once set, route-change
    /// engine rebuilds do not reset it — t=0 is the very first buffer.
    private let didNotifyFirstBuffer = Atomic<Bool>(false)

    /// Set to `true` once the real-time tap delivers a buffer for the current
    /// engine build. Cleared on each `buildAndStartEngineOrThrow`. Read by
    /// `handleConfigurationChange` (on `engineQueue`) to decide whether to
    /// absorb or honour a config-change. Written atomically from the audio
    /// thread, read on `engineQueue` — Atomic avoids any data race.
    private let currentEngineBufferDelivered = Atomic<Bool>(false)

    init(encoder: EncoderSettings = .voice) {
        self.encoder = encoder
        processingFormat = encoder.processingFormat
    }

    // MARK: - CaptureEngine conformance

    func start(writingTo url: URL) async throws {
        guard !capturingFlag.load(ordering: .acquiring) else { return }

        // New session: re-arm the one-shot first-buffer anchor. A start() can
        // legitimately run again after a *failed* start (the recorder stays
        // retryable), so this must reset — otherwise the retry never re-fires
        // the anchor and two-track alignment silently degrades. NOT reset on
        // reconnect: t=0 is the first buffer of the session, not each rebuild.
        didNotifyFirstBuffer.store(false, ordering: .releasing)

        let file = try VPIOFileHelper.createExtAudioFile(
            url: url, encoder: encoder, processingFormat: processingFormat
        )
        setExtFile(file)
        capturingFlag.store(true, ordering: .releasing)

        // Run the initial engine build on engineQueue via a continuation so
        // we don't block a cooperative thread. The build (including the ~1 s
        // output-device reclock poll) completes before start() returns, so
        // AudioRecorder can start the system engine against a stable rate.
        // Route-change rebuilds remain fire-and-forget via handleConfigurationChange.
        //
        // Install the observer after initial setup so enabling VPIO does not
        // queue recovery for an unfinished graph. Later settling notifications
        // are handled according to the current engine's running/buffer state.
        do {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                engineQueue.async { [self] in
                    isTearingDown = false
                    do {
                        try buildAndStartEngineOrThrow()
                        installConfigChangeObserver()
                        cont.resume()
                    } catch {
                        teardownEngine()
                        restoreOutputRate()
                        cont.resume(throwing: error)
                    }
                }
            }
        } catch {
            capturingFlag.store(false, ordering: .releasing)
            closeExtFile()
            throw CaptureError.micEngineFailed(
                error.localizedDescription
            )
        }
    }

    func stop() async {
        guard capturingFlag.exchange(false, ordering: .acquiringAndReleasing) else {
            return
        }

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            engineQueue.async { [self] in
                isTearingDown = true
                removeConfigChangeObserver()
                teardownEngine()
                restoreOutputRate()
                closeExtFile()
                cont.resume()
            }
        }
    }

    func reconnect() async throws {
        guard capturingFlag.load(ordering: .acquiring) else { return }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            engineQueue.async { [self] in
                guard !isTearingDown else {
                    cont.resume()
                    return
                }
                if let inID = CoreAudioHelpers.defaultInputDeviceID() {
                    let name = CoreAudioHelpers.deviceName(for: inID) ?? "unknown"
                    logger.info("Mic reconnect: following default input \"\(name)\" id=\(inID)")
                }
                teardownEngine()
                do {
                    try buildAndStartEngineOrThrow()
                    cont.resume()
                } catch {
                    logger.error("Mic reconnect failed: \(error.localizedDescription)")
                    teardownEngine()
                    restoreOutputRate()
                    closeExtFile()
                    capturingFlag.store(false, ordering: .releasing)
                    cont.resume(throwing: CaptureError.micEngineFailed(
                        error.localizedDescription
                    ))
                }
            }
        }
    }

    deinit {
        // Remove the observer first so no config-change fires during teardown.
        removeConfigChangeObserver()
        teardownEngine()
        restoreOutputRate()
        closeExtFile()
    }

    // MARK: - Engine lifecycle (engineQueue)

    /// Throwing core of the engine build (initial start + route-change).
    private func buildAndStartEngineOrThrow() throws {
        guard capturingFlag.load(ordering: .acquiring) else { return }

        // Reset the per-build buffer-delivered flag so the config-change
        // handler knows this is a fresh engine that hasn't settled yet.
        currentEngineBufferDelivered.store(false, ordering: .releasing)

        ensureOutputRateMatchesInput()

        let newEngine = AVAudioEngine()
        engine = newEngine // store early so teardownEngine() works on failure
        let input = newEngine.inputNode

        try input.setVoiceProcessingEnabled(true)
        input.voiceProcessingOtherAudioDuckingConfiguration = .init(
            enableAdvancedDucking: false, duckingLevel: .min
        )

        try installCaptureGraphAndStart(newEngine)
    }

    /// Re-query the hardware format for both initial setup and startup recovery.
    /// The existing voice-processing unit stays enabled when recovering a stopped
    /// engine, so recovery does not recreate the aggregate device that just settled.
    private func installCaptureGraphAndStart(_ engine: AVAudioEngine) throws {
        let input = engine.inputNode
        let tapFormat = input.outputFormat(forBus: 0)
        guard tapFormat.sampleRate > 0, tapFormat.channelCount > 0 else {
            throw CaptureError.micEngineFailed("The microphone has no usable audio format.")
        }
        let inputID = CoreAudioHelpers.defaultInputDeviceID() ?? 0
        let outputID = CoreAudioHelpers.defaultOutputDeviceID() ?? 0
        let inputRate = CoreAudioHelpers.nominalSampleRate(for: inputID) ?? 0
        let outputRate = CoreAudioHelpers.nominalSampleRate(for: outputID) ?? 0
        let inputName = CoreAudioHelpers.deviceName(for: inputID) ?? "unknown"
        let outputName = CoreAudioHelpers.deviceName(for: outputID) ?? "unknown"
        logger.notice("Mic setup: input=\(inputName) id=\(inputID, privacy: .public) rate=\(inputRate, privacy: .public); output=\(outputName) id=\(outputID, privacy: .public) rate=\(outputRate, privacy: .public); tap rate=\(tapFormat.sampleRate, privacy: .public) channels=\(tapFormat.channelCount, privacy: .public)")
        attachSilentOutput(to: engine, inputRate: tapFormat.sampleRate)

        input.installTap(
            onBus: 0, bufferSize: 1024, format: tapFormat
        ) { [weak self] buffer, when in
            self?.handleTap(buffer: buffer, when: when)
        }

        engine.prepare()
        try engine.start()
    }

    private func recoverEngineAfterConfigurationChange(_ current: AVAudioEngine, hasDeliveredBuffer: Bool) {
        do {
            if hasDeliveredBuffer {
                logger.notice("Config-change honoured — rebuilding mic engine after audio delivery")
                teardownEngine()
                try buildAndStartEngineOrThrow()
            } else {
                // Recreating VPIO on every pre-buffer stop repeatedly provokes the
                // same startup configuration change on some Bluetooth/display routes.
                // Keep the stopped engine and VPIO unit; refresh only our tap/output.
                logger.notice("Config-change honoured — restarting existing mic engine during startup")
                current.inputNode.removeTap(onBus: 0)
                if let node = silenceNode { current.detach(node) }
                silenceNode = nil
                cachedConverter = nil
                cachedConverterSourceHash = 0
                try installCaptureGraphAndStart(current)
                logger.notice("Mic startup restart completed: running=\(current.isRunning, privacy: .public)")
            }
        } catch {
            logger.error("Mic configuration recovery failed: \(error.localizedDescription, privacy: .public)")
            // stop() becomes a no-op once capturingFlag is cleared. Remove the
            // observer here so a later startup retry cannot leave it registered.
            removeConfigChangeObserver()
            teardownEngine()
            restoreOutputRate()
            closeExtFile()
            capturingFlag.store(false, ordering: .releasing)
            // Keep the error handler off the serial engine lifecycle queue.
            let handler = onUnrecoverableError
            DispatchQueue.global().async { handler?(error) }
        }
    }

    private func ensureOutputRateMatchesInput() {
        guard let inID = CoreAudioHelpers.defaultInputDeviceID(),
              let inRate = CoreAudioHelpers.nominalSampleRate(for: inID),
              let outID = CoreAudioHelpers.defaultOutputDeviceID(),
              let outRate = CoreAudioHelpers.nominalSampleRate(for: outID)
        else { return }
        guard abs(inRate - outRate) > 1 else { return }
        if outputRateOverride == nil {
            outputRateOverride = (outID, outRate)
        }
        let status = CoreAudioHelpers.setNominalSampleRate(inRate, for: outID)
        logger.notice("Mic output rate request: device=\(outID, privacy: .public), from=\(outRate, privacy: .public), target=\(inRate, privacy: .public), status=\(status, privacy: .public)")
        for _ in 0 ..< 40 {
            if let now = CoreAudioHelpers.nominalSampleRate(for: outID),
               abs(now - inRate) < 1
            { break }
            usleep(25000)
        }
        let rateAfter = CoreAudioHelpers.nominalSampleRate(for: outID) ?? 0
        logger.notice("Mic output rate after request: device=\(outID, privacy: .public), actual=\(rateAfter, privacy: .public)")
    }

    private func restoreOutputRate() {
        guard let override = outputRateOverride else { return }
        CoreAudioHelpers.setNominalSampleRate(override.originalRate, for: override.deviceID)
        outputRateOverride = nil
    }

    /// Connects a silent source node straight to `outputNode` (not through
    /// `mainMixerNode`) with sample rate forced to the VPIO input rate.
    private func attachSilentOutput(to engine: AVAudioEngine, inputRate: Double) {
        let output = engine.outputNode
        let hwChannels = max(1, output.outputFormat(forBus: 0).channelCount)
        guard let format = AVAudioFormat(
            standardFormatWithSampleRate: inputRate, channels: hwChannels
        ) else { return }

        let node = AVAudioSourceNode(format: format) { isSilence, _, _, bufList in
            isSilence.pointee = ObjCBool(true)
            let abl = UnsafeMutableAudioBufferListPointer(bufList)
            for buf in abl {
                if let data = buf.mData { memset(data, 0, Int(buf.mDataByteSize)) }
            }
            return noErr
        }
        engine.attach(node)
        engine.connect(node, to: output, format: format)
        silenceNode = node
    }

    /// Tears down the current engine. Crash-safe sequence based on WWDC19
    /// Session 510 ("What's New in AVAudioEngine"): VP toggling requires the
    /// engine to be in a **stopped** state. The previous code called
    /// `setVoiceProcessingEnabled(false)` while the engine was still running,
    /// which threw on every teardown — leaving VPIO enabled for dealloc.
    ///
    /// Correct order:
    ///   1. Remove the input tap (stop the real-time callback).
    ///   2. Stop the engine (required before VP can be toggled).
    ///   3. Disable voice processing (engine is stopped → succeeds).
    ///   4. Detach the silence node and nil out references.
    private func teardownEngine() {
        guard let eng = engine else {
            silenceNode = nil
            cachedConverter = nil
            cachedConverterSourceHash = 0
            return
        }

        // 1. Remove the input tap first — this stops the real-time callback
        //    from firing and prevents new writes to the file.
        eng.inputNode.removeTap(onBus: 0)

        // 2. Stop the engine. WWDC19-510: "Voice processing cannot be enabled
        //    dynamically … the engine needs to be in a stop state."
        if eng.isRunning { eng.stop() }

        // 3. Disable voice processing AFTER stop. The crash in Failure Mode A
        //    shows AVAudioEngine.dealloc hitting
        //    AUGraphNodeIOV3::DeallocateInputBlock on a node that still has
        //    VPIO enabled. Disabling VP while stopped tears down the
        //    AUVoiceProcessor graph cleanly under our control (not in dealloc).
        //    The previous code tried this before stop() — which always threw
        //    because VP toggling requires a stopped engine.
        do {
            try eng.inputNode.setVoiceProcessingEnabled(false)
            logger.notice("Teardown: voice processing disabled")
        } catch {
            // Not fatal — we're tearing down anyway, but a failure here means
            // VPIO is still enabled at dealloc, which risks the original crash.
            logger.error("Teardown: setVoiceProcessingEnabled(false) failed: \(error.localizedDescription, privacy: .public)")
        }

        // 4. Detach the silence node and nil out references.
        if let node = silenceNode { eng.detach(node) }
        silenceNode = nil
        engine = nil
        cachedConverter = nil
        cachedConverterSourceHash = 0
    }

    // MARK: - Config-change observer

    /// Installs the config-change observer. Must run on `engineQueue`
    /// (same as `removeConfigChangeObserver`) so `configObserver` is
    /// accessed from a single serial context.
    private func installConfigChangeObserver() {
        // Follow rebuilt engines, but ignore queued notifications from an old
        // graph or another AVAudioEngine in this process.
        configObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: nil, queue: nil
        ) { [weak self] notification in
            guard let source = notification.object as? AVAudioEngine else { return }
            self?.handleConfigurationChange(engineID: ObjectIdentifier(source))
        }
    }

    /// Removes the config-change observer. Must run on `engineQueue`
    /// (same as `installConfigChangeObserver`) so `configObserver` is
    /// accessed from a single serial context. Exception: `deinit`, where
    /// no concurrent access is possible.
    private func removeConfigChangeObserver() {
        if let observer = configObserver {
            NotificationCenter.default.removeObserver(observer)
            configObserver = nil
        }
    }

    /// A configuration notification may arrive after VPIO starts but before its
    /// first buffer. Ignore it only if the current engine is still running. If it
    /// stopped, restart the existing graph without toggling voice processing;
    /// recreating that unit can reproduce the startup change indefinitely.
    /// AudioRecorder independently bounds the wait for the first written buffer.
    private func handleConfigurationChange(engineID: ObjectIdentifier) {
        engineQueue.async { [weak self] in
            guard let self, !isTearingDown, capturingFlag.load(ordering: .acquiring),
                  let engine, ObjectIdentifier(engine) == engineID else { return }

            let settled = currentEngineBufferDelivered.load(ordering: .acquiring)
            let running = engine.isRunning
            logger.notice("Mic config change: running=\(running, privacy: .public), bufferDelivered=\(settled, privacy: .public)")

            if !settled, running {
                // Absorb: this is the VPIO startup-settle config change.
                logger.info("Config-change absorbed during startup settle (no buffer yet)")
                return
            }

            recoverEngineAfterConfigurationChange(engine, hasDeliveredBuffer: settled)
        }
    }

    // MARK: - Tap (real-time audio thread)

    private func handleTap(buffer: AVAudioPCMBuffer, when: AVAudioTime) {
        guard buffer.frameLength > 0 else { return }
        // Mark that this engine build has delivered a buffer. This arms the
        // config-change handler to honour subsequent notifications (the
        // startup-settle window is over). The store is idempotent after the
        // first buffer. The `.releasing` store pairs with the `.acquiring`
        // load in `handleConfigurationChange` for a proper release/acquire
        // edge. A stopped engine must be recovered even if this flag is false;
        // configuration notifications are not guaranteed to repeat.
        if !currentEngineBufferDelivered.load(ordering: .relaxed) {
            currentEngineBufferDelivered.store(true, ordering: .releasing)
        }

        guard let mono = VPIOBufferHelper.extractChannel0(buffer) else { return }

        let targetFormat = processingFormat
        let bufferToWrite: AVAudioPCMBuffer
        if mono.format.sampleRate == targetFormat.sampleRate {
            bufferToWrite = mono
        } else {
            guard let converter = converterForSource(mono.format),
                  let converted = VPIOBufferHelper.convert(
                      mono, to: targetFormat, using: converter
                  )
            else { return }
            bufferToWrite = converted
        }

        // Serialize the file write against closeExtFile()'s dispose. `trylock`
        // (never block) on the real-time thread: if teardown holds the lock we
        // simply drop this buffer — we're stopping anyway, and a write into a
        // disposed AAC encoder would corrupt the heap. The file ref is loaded
        // *inside* the lock so it can't be disposed between load and write.
        guard os_unfair_lock_trylock(&_fileLock) else { return }
        defer { os_unfair_lock_unlock(&_fileLock) }
        guard let file = currentExtFile() else { return }
        if VPIOBufferHelper.writeBuffer(bufferToWrite, to: file) == noErr {
            notifyFirstBufferIfNeeded(when)
        }
    }

    /// Fires `onFirstBuffer` exactly once with the host-clock seconds of the
    /// first delivered buffer. Derives the anchor from `when.hostTime` via
    /// `AudioConvertHostTimeToNanos` -- the same clock base the system engine
    /// uses to pad the system track, so the two stay aligned.
    private func notifyFirstBufferIfNeeded(_ when: AVAudioTime) {
        guard !didNotifyFirstBuffer.exchange(true, ordering: .acquiringAndReleasing) else { return }
        let anchor: Double = if when.isHostTimeValid {
            Double(AudioConvertHostTimeToNanos(when.hostTime)) / 1_000_000_000
        } else {
            0
        }
        logger.notice("First mic buffer delivered -- anchor=\(anchor, privacy: .public)s")
        onFirstBuffer?(anchor)
    }

    private func converterForSource(_ sourceFormat: AVAudioFormat) -> AVAudioConverter? {
        let sourceHash = sourceFormat.hash
        if sourceHash == cachedConverterSourceHash, let converter = cachedConverter {
            return converter
        }
        guard let converter = AVAudioConverter(
            from: sourceFormat, to: processingFormat
        ) else {
            logger.error("Failed to build AVAudioConverter for mic resampling")
            return nil
        }
        cachedConverter = converter
        cachedConverterSourceHash = sourceHash
        return converter
    }
}

// MARK: - File handle (lock-free, safe for real-time thread)

extension LiveMicCaptureEngine {
    private func setExtFile(_ file: ExtAudioFileRef?) {
        let bits = file.map { UInt(bitPattern: $0) } ?? 0
        atomicFileRef.store(bits, ordering: .releasing)
    }

    private func closeExtFile() {
        // Take the lock so any in-flight tap write completes before dispose
        // (see `_fileLock`). Zero the ref inside the lock so a tap that has
        // not yet taken the lock observes nil and skips its write.
        os_unfair_lock_lock(&_fileLock)
        defer { os_unfair_lock_unlock(&_fileLock) }
        let bits = atomicFileRef.exchange(0, ordering: .acquiringAndReleasing)
        if bits != 0, let ptr = OpaquePointer(bitPattern: bits) {
            ExtAudioFileDispose(ptr)
        }
    }

    private func currentExtFile() -> ExtAudioFileRef? {
        let bits = atomicFileRef.load(ordering: .acquiring)
        guard bits != 0 else { return nil }
        return OpaquePointer(bitPattern: bits)
    }
}

// MARK: - Buffer helpers (pure, testable)

/// Pure audio-buffer operations used by the VPIO mic tap. Extracted from the
/// engine class so they can be unit-tested without hardware.
enum VPIOBufferHelper {
    /// Extracts channel 0 of a multichannel non-interleaved float PCM buffer
    /// into a mono buffer at the same sample rate. With VPIO, channel 0 is
    /// the processed/beamformed mono; the rest are raw-array reference feeds.
    static func extractChannel0(_ source: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        let frames = Int(source.frameLength)
        guard frames > 0, let srcData = source.floatChannelData else { return nil }
        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: source.format.sampleRate,
            channels: 1,
            interleaved: false
        ), let mono = AVAudioPCMBuffer(
            pcmFormat: monoFormat,
            frameCapacity: AVAudioFrameCount(frames)
        ) else { return nil }
        mono.frameLength = AVAudioFrameCount(frames)
        guard let dst = mono.floatChannelData?[0] else { return nil }
        dst.update(from: srcData[0], count: frames)
        return mono
    }

    /// Resamples a PCM buffer to `targetFormat` via the supplied converter.
    static func convert(
        _ source: AVAudioPCMBuffer,
        to targetFormat: AVAudioFormat,
        using converter: AVAudioConverter
    ) -> AVAudioPCMBuffer? {
        let frameCapacity = AVAudioFrameCount(
            Double(source.frameLength) * targetFormat.sampleRate / source.format.sampleRate
        )
        guard frameCapacity > 0,
              let output = AVAudioPCMBuffer(
                  pcmFormat: targetFormat, frameCapacity: frameCapacity
              )
        else { return nil }

        var error: NSError?
        nonisolated(unsafe) var hasProvidedData = false
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            if hasProvidedData { outStatus.pointee = .noDataNow; return nil }
            hasProvidedData = true
            outStatus.pointee = .haveData
            return source
        }
        converter.convert(to: output, error: &error, withInputFrom: inputBlock)
        if let error {
            logger.error("AVAudioConverter error: \(error.localizedDescription)")
            return nil
        }
        guard output.frameLength > 0 else { return nil }
        return output
    }

    @discardableResult
    static func writeBuffer(
        _ buffer: AVAudioPCMBuffer, to file: ExtAudioFileRef
    ) -> OSStatus {
        let status = ExtAudioFileWrite(
            file, buffer.frameLength, buffer.mutableAudioBufferList
        )
        if status != noErr {
            logger.error("ExtAudioFileWrite error: \(status)")
        }
        return status
    }
}
