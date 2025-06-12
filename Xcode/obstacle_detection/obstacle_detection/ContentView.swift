import SwiftUI
import Speech
import AVFoundation

struct ContentView: View {
    @State private var navigateToARView = false
    @State private var navigateToGlassView = false
    //@State private var isGlassViewPresented = false  // isPresented 用の State 追加
    @StateObject private var speechRecognizer = SpeechRecognizer()
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.white.ignoresSafeArea()
                VStack {
                    Text("ボタンを押してください")
                        .font(.title)
                        .foregroundColor(.black)
                        .padding()
                    
                    HStack(spacing: 20) {
                        Spacer()
                        
                        Button(action: {
                            speechRecognizer.stopListening()
                            navigateToARView = true
                        }) {
                            ModeButtonView(imageName: "earphone", text: "骨伝導\nイヤホン\nモード", color: .green)
                        }
                        
                        Button(action: {
                            speechRecognizer.stopListening()
                            navigateToGlassView = true
                        }) {
                            ModeButtonView(imageName: "ar", text: "グラスモード", color: .blue)
                        }
                        
                        Spacer()
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                }
            }
            .navigationDestination(isPresented: $navigateToARView) {
                ARView()
            }
            .navigationDestination(isPresented: $navigateToGlassView) {
                //GlassView(isPresented: $isGlassViewPresented, navigateToGlassView: $navigateToGlassView)
                GlassView()
            }
        }
        .onAppear {
            speechRecognizer.startAppMessage()
            speechRecognizer.startListening { recognizedText in
                if recognizedText.contains("イヤホンモード開始") {
                    navigateToARView = true
                    speechRecognizer.stopListening()
                } else if recognizedText.contains("グラスモード開始") || recognizedText.contains("クラスモード開始") {
                    navigateToGlassView = true
                    speechRecognizer.stopListening()
                }
            }
        }
        .onDisappear {
            speechRecognizer.stopListening()
        }
    }
}

struct ModeButtonView: View {
    let imageName: String
    let text: String
    let color: Color
    
    var body: some View {
        VStack {
            Image(imageName)
                .resizable()
                .scaledToFit()
                .frame(width: 140, height: 300)
            Text(text)
                .font(.system(size: 24))
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
        }
        .padding()
        .frame(width: 165, height: 500)
        .background(color)
        .cornerRadius(10)
    }
}

// 音声認識 & 音声合成を管理するクラス
class SpeechRecognizer: ObservableObject {
    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "ja-JP"))
    private let speechSynthesizer = AVSpeechSynthesizer()
    private var recognitionTask: SFSpeechRecognitionTask?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var audioEngine = AVAudioEngine()
    
    func startAppMessage() {
        speak("階段・段差検知アプリです。イヤホンを使いたいかたはイヤホンモード開始、ARグラスを使いたいかたはグラスモード開始と言ってください。")
    }
    
    func startListening(completion: @escaping (String) -> Void) {
        SFSpeechRecognizer.requestAuthorization { status in
            DispatchQueue.main.async {
                if status == .authorized {
                    self.configureAudioSession()
                    self.recognitionTask?.cancel()
                    self.recognitionTask = nil
                    self.recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
                    
                    guard let recognitionRequest = self.recognitionRequest else { return }
                    let inputNode = self.audioEngine.inputNode
                    inputNode.removeTap(onBus: 0)
                    
                    let recordingFormat = inputNode.outputFormat(forBus: 0)
                    inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
                        recognitionRequest.append(buffer)
                    }
                    
                    self.audioEngine.prepare()
                    do {
                        try self.audioEngine.start()
                    } catch {
                        print("音声認識の開始に失敗しました: \(error)")
                        return
                    }
                    
                    self.recognitionTask = self.recognizer?.recognitionTask(with: recognitionRequest) { result, error in
                        guard let result = result else {
                            if let error = error {
                                print("音声認識エラー: \(error)")
                            }
                            return
                        }
                        let spokenText = result.bestTranscription.formattedString
                        print("認識したテキスト: \(spokenText)")
                        completion(spokenText)
                    }
                } else {
                    print("音声認識の権限がありません")
                }
            }
        }
    }
    
    func stopListening() {
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
            recognitionRequest?.endAudio()
            recognitionTask?.cancel()
        }
    }
    
    private func configureAudioSession() {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers, .allowBluetooth, .allowBluetoothA2DP])
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("AVAudioSession の設定に失敗しました: \(error)")
        }
    }
    
    private func speak(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "ja-JP")
        utterance.rate = 0.5
        speechSynthesizer.speak(utterance)
    }
}
