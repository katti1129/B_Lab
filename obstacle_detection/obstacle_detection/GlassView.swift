//
//  Untitled.swift
//  障害物検知
//
//  Created by かっち on 2024/12/09.
//
import SwiftUI
import UIKit
import AVFoundation
import Vision


// 次の画面1: ARモード画面
struct GlassView: View {
    @Environment(\.dismiss) var dismiss // 戻る機能を実装するための環境変数
    //@Environment(\.presentationMode) var presentationMode
    //@Binding var isPresented: Bool
    //@Binding var navigateToGlassView: Bool
    //let synthesizer = AVSpeechSynthesizer()
    var body: some View {
        VStack {
            CameraPreview(viewControllerType: GlassViewController.self)
                .edgesIgnoringSafeArea(.all)
            // 戻るボタンを画面下部に配置
            Button(action: {
                dismiss() // 戻るアクション
                //isPresented = false // NavigationStackから抜ける
            }) {
                Text("戻る")
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.blue)
                    .cornerRadius(10)
                    .padding(.horizontal)
            }
            .padding(.bottom, 20) // 余白を追加してボタンを画面下部に配置
        }
        .navigationBarBackButtonHidden(true)
    }
}




// YOLOによる検出処理を行うViewController
class GlassViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    var confidenceLabel: UILabel! // 信頼値表示用のラベル
    var captureSession = AVCaptureSession()
    var previewView = UIImageView()
    var previewLayer:AVCaptureVideoPreviewLayer!
    var videoOutput: AVCaptureVideoDataOutput!
    var frameCounter = 0
    var frameInterval = 1
    var videoSize = CGSize.zero
    let ciContext = CIContext()
    var classes: [String] = []// クラスラベルとモデル
    var speechSynthesizer = AVSpeechSynthesizer()
    var hasSpokenWarning = false
    var warningLabel: UILabel!// 警告用ラベル
    var detectionLabel: UILabel!  // 検出結果表示用のラベル
    var canSpeakWarning = true // 音声発生の制御用フラグ
    
    lazy var yoloRequest: VNCoreMLRequest! = {
        do {
            let model = try best().model
            guard let classes = model.modelDescription.classLabels as? [String] else {
                fatalError("モデルクラスの読み込みに失敗しました")
            }
            self.classes = classes
            let vnModel = try VNCoreMLModel(for: model)
            let request = VNCoreMLRequest(model: vnModel)
            return request
        } catch {
            fatalError("mlmodelの初期化エラー")
        }
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        //view.backg
        setupVideo()
        setupWarningLabel()
        setupConfidenceLabel() // 信頼値表示用ラベルの設定
        setupDetectionLabel()
        
    }
    
    // 信頼値表示用のラベルを設定
    func setupConfidenceLabel() {
        confidenceLabel = UILabel()
        confidenceLabel.textColor = .white
        //confidenceLabel.backgroundColor = .black//信頼値表示部分のみ黒
        confidenceLabel.font = UIFont.boldSystemFont(ofSize: 30)
        confidenceLabel.textAlignment = .left
        confidenceLabel.frame = CGRect(x: 0, y: 40, width: 500, height: 100) // 左上に配置
        confidenceLabel.text = "信頼値: - "
        //confidenceLabel.text = "信頼値: - 距離: -" // 初期テキスト
        view.addSubview(confidenceLabel)
    }
    
    func setupWarningLabel() {
        let warningLabel = UILabel()
        warningLabel.textColor = .white
        warningLabel.backgroundColor = .red
        warningLabel.font = UIFont.boldSystemFont(ofSize: 24)
        warningLabel.textAlignment = .center
        warningLabel.frame = CGRect(x: 0, y: 150, width: view.bounds.width, height: 50)
        warningLabel.isHidden = true
        view.addSubview(warningLabel)
        self.warningLabel = warningLabel
    }
    
    func setupDetectionLabel(){
        detectionLabel = UILabel()
        detectionLabel.textColor = .white
        //detectionLabel.backgroundColor = .black//信頼値表示部分のみ黒
        detectionLabel.font = UIFont.boldSystemFont(ofSize: 30)
        detectionLabel.textAlignment = .left
        detectionLabel.frame = CGRect(x: 0, y: 70, width: 500, height: 100) // 左上に配置
        detectionLabel.text = "物体: - "
        //detectionLabel.text = "信頼値: - 距離: -" // 初期テキスト
        view.addSubview(detectionLabel)
    }
    
    func setupVideo() {
        previewView.frame = view.bounds
        view.addSubview(previewView)
        
        // 黒色の背景ビューを作成
        let redBackgroundView = UIView(frame: view.bounds)
        redBackgroundView.backgroundColor = UIColor.black
        view.addSubview(redBackgroundView) // 最背面に追加
        
        captureSession.beginConfiguration()
        
        let device = AVCaptureDevice.default(for: AVMediaType.video)
        let deviceInput = try! AVCaptureDeviceInput(device: device!)
        
        captureSession.addInput(deviceInput)
        videoOutput = AVCaptureVideoDataOutput()
        
        let queue = DispatchQueue(label: "VideoQueue")
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        captureSession.addOutput(videoOutput)
        if let videoConnection = videoOutput.connection(with: .video) {
            if videoConnection.isVideoOrientationSupported {
                videoConnection.videoOrientation = .portrait
            }
        }
        captureSession.commitConfiguration()
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        
        previewLayer?.frame = previewView.bounds
        previewLayer.videoGravity = .resizeAspectFill // アスペクト比を維持しつつ画面全体を埋める
        previewLayer?.connection?.videoOrientation = AVCaptureVideoOrientation.portrait
        previewView.layer.addSublayer(previewLayer!)
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }
    
    func detection(pixelBuffer: CVPixelBuffer) {
        do {
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
            try handler.perform([yoloRequest])
            //VNRecognizedObjectObservationでresult.confidence取得可能になる
            guard let results = yoloRequest.results as? [VNRecognizedObjectObservation] else {
                return
            }
            
            for result in results {
                if let label = result.labels.first?.identifier {
                    DispatchQueue.main.async {
                        // 信頼値を左上に表示(例えば、result.confidence が 0.8532 である場合、String(format: "信頼値: %.2f", result.confidence) は "信頼値: 0.85" という文字列を生成します。)
                        
                        self.confidenceLabel.text = String(format: "信頼値 : (%.2f%%)",result.confidence*100)// 信頼値を更新
                        self.detectionLabel.text = String(format: "物体 : %@", label)// 物体を更新
                        
                        
                        if label == "upstair" && result.confidence >= 0.8{
                            self.detectionLabel.textColor = .red
                            self.warningLabel.text = "上り階段を発見しました。"
                            self.warningLabel.isHidden = false
                            if !self.hasSpokenWarning && self.canSpeakWarning{
                                self.speakWarning(message: "上り階段を発見しました。")
                                self.hasSpokenWarning = true
                            }
                        } else if label == "downstair" && result.confidence >= 0.8{
                            self.detectionLabel.textColor = .blue
                            self.warningLabel.text = "下り階段を発見しました。"
                            self.warningLabel.isHidden = false
                            if !self.hasSpokenWarning && self.canSpeakWarning{
                                self.speakWarning(message: "下り階段を発見しました。")
                                self.hasSpokenWarning = true
                            }
                        }else if label == "step" && result.confidence >= 0.6{
                            self.detectionLabel.textColor = .green
                            self.warningLabel.text = "段差を発見しました"
                            self.warningLabel.isHidden = false
                            if !self.hasSpokenWarning && self.canSpeakWarning{
                                self.speakWarning(message: "段差を発見しました。")
                                self.hasSpokenWarning = true
                            }
                        }
                    }
                    return
                }
            }
            DispatchQueue.main.async {
                self.warningLabel.isHidden = true
                self.hasSpokenWarning = false // ラベルが表示されない場合、警告フラグをリセット
            }
        } catch {
            print(error)
        }
    }
    
    func speakWarning(message: String) {
        let utterance = AVSpeechUtterance(string: message)
        utterance.voice = AVSpeechSynthesisVoice(language: "ja-JP")
        speechSynthesizer.speak(utterance)
        hasSpokenWarning = true
        canSpeakWarning = false // 音声発生を抑制
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
            self.canSpeakWarning = true // 5秒後に音声発生を再度許可
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        frameCounter += 1
        if videoSize == CGSize.zero {
            guard let width = sampleBuffer.formatDescription?.dimensions.width,
                  let height = sampleBuffer.formatDescription?.dimensions.height else {
                fatalError()
            }
            videoSize = CGSize(width: CGFloat(width), height: CGFloat(height))
        }
        if frameCounter == frameInterval {
            frameCounter = 0
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            detection(pixelBuffer: pixelBuffer)
        }
    }
}




