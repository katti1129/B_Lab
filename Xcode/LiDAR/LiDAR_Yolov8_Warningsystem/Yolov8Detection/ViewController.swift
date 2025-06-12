//
//  ViewController.swift
//  Yolov8Detection
//
//  Created by かっち on 2024/07/19.
//

//
//  ViewController.swift
//  LiDAR
//
//  Created by かっち on 2024/10/03.
//


import UIKit
import ARKit
import Vision

class ViewController: UIViewController, ARSessionDelegate {

    var sceneView: ARSCNView!
    var depthLabel: UILabel!
    var depthRectangle: UIView!
    var detectionLabel: UILabel!  // 検出結果表示用のラベル
    var warningLabel: UILabel!

    // YOLOのリクエストを保持するプロパティ
    lazy var yoloRequest: VNCoreMLRequest = {
        do {
            let model = try VNCoreMLModel(for: best().model)
            return VNCoreMLRequest(model: model, completionHandler: self.handleYOLOResults)
        } catch {
            fatalError("Failed to load YOLO model: \(error)")
        }
    }()

    override func viewDidLoad() {
        super.viewDidLoad()

        // ARSCNViewをセットアップ
        sceneView = ARSCNView(frame: self.view.frame)
        self.view.addSubview(sceneView)

        // ARセッションのデリゲートを設定
        sceneView.session.delegate = self

        // ARセッションの設定
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = .sceneDepth
        sceneView.session.run(configuration)

        // 深度情報を表示するラベルをセットアップ
        depthLabel = UILabel(frame: CGRect(x: 20, y: 0, width: 200, height: 50))
        depthLabel.textColor = .white
        depthLabel.font = UIFont.systemFont(ofSize: 24, weight: .semibold)
        depthLabel.text = "Depth: --"
        self.view.addSubview(depthLabel)
        
        // 検出結果を表示するラベルをセットアップ
        detectionLabel = UILabel(frame: CGRect(x: 400, y: 0, width: 300, height: 50))
        detectionLabel.textColor = .white
        detectionLabel.font = UIFont.systemFont(ofSize: 24, weight: .semibold)
        detectionLabel.text = "Detection: --"
        self.view.addSubview(detectionLabel)
        
        // warningLabelの設定
        warningLabel = UILabel()
        warningLabel.translatesAutoresizingMaskIntoConstraints = false
        warningLabel.textColor = .white  // 好みで色を設定
        warningLabel.font = UIFont.systemFont(ofSize: 36, weight: .semibold)
        
        warningLabel.backgroundColor = .red  // 背景色
        warningLabel.layer.borderColor = UIColor.red.cgColor  // 枠線の色
        warningLabel.layer.borderWidth = 2.0  // 枠線の幅
        warningLabel.layer.cornerRadius = 8.0  // 角丸
        warningLabel.layer.masksToBounds = true  // 枠線内にクリップ
        self.view.addSubview(warningLabel)
        
        // warningLabelのAuto Layoutの設定
        NSLayoutConstraint.activate([
            warningLabel.centerXAnchor.constraint(equalTo: self.view.centerXAnchor),
            warningLabel.bottomAnchor.constraint(equalTo: self.view.bottomAnchor, constant: -20)
        ])
        

        // 深度領域を示す四角形をセットアップ
        /*
        depthRectangle = UIView(frame: CGRect(x: self.view.frame.midX - 50, y: self.view.frame.midY - 50, width: 100, height: 100))
        depthRectangle.layer.borderWidth = 2
        depthRectangle.layer.borderColor = UIColor.red.cgColor
        self.view.addSubview(depthRectangle)
         */
    }

    // セッションでフレームが更新されるたびに呼ばれる
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let sceneDepth = frame.sceneDepth else { return }

        // 深度データを処理（これは毎フレーム実行）
        self.updateDepthInformation(from: frame)

        // YOLOの推論は非同期でバックグラウンドスレッドで実行
        DispatchQueue.global(qos: .userInitiated).async {
            self.performYOLODetection(on: frame)
        }
    }

    // 深度情報を更新する関数
    func updateDepthInformation(from frame: ARFrame) {
        let depthMap = frame.sceneDepth!.depthMap
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let baseAddress = CVPixelBufferGetBaseAddress(depthMap)
        let rowBytes = CVPixelBufferGetBytesPerRow(depthMap)

        let centerX = width / 2
        let centerY = height / 2
        let regionSize = 3
        var totalDepth: Float32 = 0
        var count = 0

        for y in (centerY - regionSize / 2)..<(centerY + regionSize / 2) {
            let rowPointer = baseAddress! + y * rowBytes
            for x in (centerX - regionSize / 2)..<(centerX + regionSize / 2) {
                let depthValue = rowPointer.assumingMemoryBound(to: Float32.self)[x]
                totalDepth += depthValue
                count += 1
            }
        }

        let averageDepth = totalDepth / Float32(count)

        DispatchQueue.main.async {
            self.depthLabel.text = String(format: "Depth: %.2f m", averageDepth)
            /*
            if averageDepth < 1.0 {
                self.depthRectangle.layer.borderColor = UIColor.red.cgColor
            } else if averageDepth < 2.0 {
                self.depthRectangle.layer.borderColor = UIColor.yellow.cgColor
            } else {
                self.depthRectangle.layer.borderColor = UIColor.blue.cgColor
            }
            */
        }
    }

    // YOLO推論を行う関数
    func performYOLODetection(on frame: ARFrame) {
        let pixelBuffer = frame.capturedImage
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try handler.perform([yoloRequest])
        } catch {
            print("YOLO detection failed: \(error)")
        }
    }

    // YOLOの推論結果を処理する関数
    func handleYOLOResults(request: VNRequest, error: Error?) {
        guard let results = request.results as? [VNRecognizedObjectObservation] else { return }

        DispatchQueue.main.async {
            /*
            for result in results {
                let label = result.labels.first?.identifier ?? "Unknown"
                let confidence = result.confidence*100
                print("Detected object: \(label) with confidence: \(confidence)")
                // ここでUIに結果を反映する処理を追加
            }
            */
            
            if let firstResult = results.first {
                let label = firstResult.labels.first?.identifier ?? "Unknown"
                let confidence = firstResult.confidence * 100
                self.detectionLabel.text = String(format: "Detected: %@ (%.2f%%)", label, confidence)
 
                
                // ラベルに応じて文字色を変更
                switch label {
                case "upstair":
                    self.detectionLabel.textColor = .red
                    if firstResult.confidence>=0.8{
                        self.warningLabel.text = "Warning: 上り階段を発見しました"
                    }
                case "downstair":
                    self.detectionLabel.textColor = .blue
                    if firstResult.confidence>=0.8{
                        self.warningLabel.text = "Warning: 下り階段を発見しました"
                    }
                default:
                    self.detectionLabel.textColor = .white
                    self.warningLabel.text = ""
                }
                
            }else {
                self.detectionLabel.text = "No objects detected"
                self.detectionLabel.textColor = .white
                self.warningLabel.text = ""
            }
        }
    }
    
}

/*
import UIKit
import AVFoundation
import Vision
import ARKit

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate, AVCapturePhotoCaptureDelegate, ARSCNViewDelegate {

    var captureSession = AVCaptureSession()
    var previewView = UIImageView()
    var previewLayer: AVCaptureVideoPreviewLayer!
    var videoOutput: AVCaptureVideoDataOutput!
    var frameCounter = 0
    var frameInterval = 1
    var videoSize = CGSize.zero
    var arView: ARSCNView!

    let ciContext = CIContext()
    var classes: [String] = []
    
    lazy var yoloRequest: VNCoreMLRequest! = {
        do {
            let model = try best().model
            print("Model loaded successfully")
            guard let classes = model.modelDescription.classLabels as? [String] else {
                fatalError()
            }
            self.classes = classes
            let vnModel = try VNCoreMLModel(for: model)
            let request = VNCoreMLRequest(model: vnModel)
            return request
        } catch {
            fatalError("mlmodel error.")
        }
    }()
    
    // 警告メッセージ用のUILabelを追加
    var warningLabel: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()
        setupARView()
        setupVideo()
        setupWarningLabel()
    }
    
    func setupARView() {
        arView = ARSCNView(frame: view.bounds)
        arView.delegate = self
        view.addSubview(arView)
        
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        configuration.frameSemantics = .sceneDepth // 深度情報を取得
        arView.session.run(configuration)
    }

    func setupWarningLabel() {
        let labelHeight: CGFloat = 50
        warningLabel = UILabel(frame: CGRect(x: 0, y: 0, width: view.bounds.width, height: labelHeight))
        warningLabel.backgroundColor = UIColor.red.withAlphaComponent(0.7)
        warningLabel.textColor = UIColor.white
        warningLabel.textAlignment = .center
        warningLabel.font = UIFont.boldSystemFont(ofSize: 20)
        warningLabel.text = "上り階段を発見しました．"
        warningLabel.isHidden = true
        view.addSubview(warningLabel)
    }

    func setupVideo() {
        previewView.frame = view.bounds
        view.addSubview(previewView)

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
                videoConnection.videoOrientation = .landscapeRight
            }
        }
        captureSession.commitConfiguration()

        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }

    func detection(pixelBuffer: CVPixelBuffer) {
        do {
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
            try handler.perform([yoloRequest])
            guard let results = yoloRequest.results as? [VNRecognizedObjectObservation] else {
                return
            }
            var upstairDetected = false

            for result in results {
                let label = result.labels.first?.identifier
                if label == "upstair" && result.confidence >= 0.85 {
                    upstairDetected = true
                    break
                }
            }

            DispatchQueue.main.async {
                self.warningLabel.isHidden = !upstairDetected
                if upstairDetected {
                    self.measureDistance()
                }
            }
        } catch {
            print(error)
        }
    }

    func measureDistance() {
        guard let frame = arView.session.currentFrame else { return }
        if let depthData = frame.capturedDepthData {
            print("Depth data available")
            let depthMap = depthData.depthDataMap
            // 特定の位置の深度データを取得
            let pixelBuffer = depthMap as CVPixelBuffer
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

            let x = Int(width / 2) // 画面の中央を指定
            let y = Int(height / 2) // 画面の中央を指定
            let offset = y * bytesPerRow + x * 2 // 2バイトごとに深度値が格納される

            let distance: Float = baseAddress?.load(fromByteOffset: offset, as: Float.self) ?? 0.0

            // 距離の表示
            print("Distance to object: \(distance) meters")
        }else{
            print("No depth data available")
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

struct Detection {
    let box: CGRect
    let confidence: Float
    let label: String?
}
 */


//iPhoneとの距離
/*
import UIKit
import ARKit

class ViewController: UIViewController, ARSessionDelegate {

    var sceneView: ARSCNView!
    var depthLabel: UILabel!
    var depthRectangle: UIView!

    override func viewDidLoad() {
        super.viewDidLoad()

        // ARSCNViewをセットアップ
        sceneView = ARSCNView(frame: self.view.frame)
        self.view.addSubview(sceneView)

        // ARセッションのデリゲートを設定
        sceneView.session.delegate = self

        // ARセッションの設定
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = .sceneDepth
        sceneView.session.run(configuration)

        // 深度情報を表示するラベルをセットアップ
        depthLabel = UILabel(frame: CGRect(x: 20, y: 50, width: 200, height: 50))
        depthLabel.textColor = .white
        depthLabel.font = UIFont.systemFont(ofSize: 24)
        depthLabel.text = "Depth: --"
        self.view.addSubview(depthLabel)

        // 深度領域を示す四角形をセットアップ
        depthRectangle = UIView(frame: CGRect(x: self.view.frame.midX - 50, y: self.view.frame.midY - 50, width: 100, height: 100))
        depthRectangle.layer.borderWidth = 2
        depthRectangle.layer.borderColor = UIColor.red.cgColor
        self.view.addSubview(depthRectangle)
    }

    // セッションでフレームが更新されるたびに呼ばれる
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let sceneDepth = frame.sceneDepth else { return }

        // 深度データを取得
        let depthMap = sceneDepth.depthMap

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        // 深度データの幅と高さを取得
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let baseAddress = CVPixelBufferGetBaseAddress(depthMap)
        let rowBytes = CVPixelBufferGetBytesPerRow(depthMap)

        // 真ん中のピクセルの座標（例: 3x3の範囲で計測）
        let centerX = width / 2
        let centerY = height / 2
        let regionSize = 3

        // 中心から3x3の範囲で計測する
        var totalDepth: Float32 = 0
        var count = 0
        for y in (centerY - regionSize / 2)..<(centerY + regionSize / 2) {
            let rowPointer = baseAddress! + y * rowBytes
            for x in (centerX - regionSize / 2)..<(centerX + regionSize / 2) {
                let depthValue = rowPointer.assumingMemoryBound(to: Float32.self)[x]
                totalDepth += depthValue
                count += 1
            }
        }

        // 平均深度を計算
        let averageDepth = totalDepth / Float32(count)

        // 深度情報をラベルに表示
        DispatchQueue.main.async {
            self.depthLabel.text = String(format: "Depth: %.2f meters", averageDepth)

            // 四角形の色を深度に基づいて変更
            if averageDepth < 1.0 {
                self.depthRectangle.layer.borderColor = UIColor.red.cgColor
            } else if averageDepth < 2.0 {
                self.depthRectangle.layer.borderColor = UIColor.yellow.cgColor
            } else {
                self.depthRectangle.layer.borderColor = UIColor.blue.cgColor
            }
        }
    }
}
 */

/*
import UIKit
import ARKit

class ViewController: UIViewController, ARSessionDelegate {
    
    var session: ARSession!
    var warningLabel: UILabel!
    @IBOutlet weak var sceneView: ARSCNView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupARSession()
        setupWarningLabel()
    }
    
    func setupARSession() {
        // ARセッションのセットアップ
        session = ARSession()
        session.delegate = self
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = .sceneDepth  // 深度データの取得を有効化
        session.run(configuration)
    }
    
    func setupWarningLabel() {
        // 警告ラベルのセットアップ
        warningLabel = UILabel(frame: CGRect(x: 0, y: 50, width: view.frame.width, height: 50))
        warningLabel.backgroundColor = .red
        warningLabel.textColor = .white
        warningLabel.textAlignment = .center
        warningLabel.text = "Warning: Steep Step Ahead!"
        warningLabel.isHidden = true
        view.addSubview(warningLabel)
    }
    
    // ARセッションのフレームごとに呼ばれるデリゲートメソッド
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let depthData = frame.capturedDepthData else { return }
        detectStep(depthData: depthData)
    }
    
    // 深度情報を用いて段差を検出するメソッド
    func detectStep(depthData: AVDepthData) {
        let depthMap = depthData.depthDataMap
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let baseAddress = CVPixelBufferGetBaseAddress(depthMap)!
        let rowBytes = CVPixelBufferGetBytesPerRow(depthMap)
        
        // 中心付近の深度値を取得
        let centerX = width / 2
        let centerY = height / 2
        let rowPointer = baseAddress + centerY * rowBytes
        let centerDepth = rowPointer.assumingMemoryBound(to: Float32.self)[centerX]
        
        // 深度値が0.5メートル以下かどうかを確認
        if centerDepth <= 0.5 {
            DispatchQueue.main.async {
                self.warningLabel.isHidden = false
            }
        } else {
            DispatchQueue.main.async {
                self.warningLabel.isHidden = true
            }
        }
    }
}
*/


/*
import UIKit
import ARKit

class ViewController: UIViewController, ARSCNViewDelegate {

    var sceneView: ARSCNView!
    var measurementLabel: UILabel!
    var lastHeight: Float = 0.0

    override func viewDidLoad() {
        super.viewDidLoad()

        // ARSCNViewのセットアップ
        sceneView = ARSCNView(frame: self.view.frame)
        sceneView.delegate = self
        self.view.addSubview(sceneView)

        // ラベルを追加して測定結果を表示
        measurementLabel = UILabel(frame: CGRect(x: 20, y: 50, width: 300, height: 50))
        measurementLabel.textColor = .white
        measurementLabel.font = UIFont.boldSystemFont(ofSize: 24)
        self.view.addSubview(measurementLabel)

        // ARセッションの設定
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal]
        sceneView.session.run(configuration)
    }

    // ARSCNViewDelegate: ノードが追加されたときに呼び出される
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        if let planeAnchor = anchor as? ARPlaneAnchor {
            updateMeasurement(for: planeAnchor)
        }
    }

    // ARセッションが更新されるたびに呼び出される
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if let hitTestResult = sceneView.hitTest(view.center, types: .featurePoint).first {
            let position = SCNVector3(hitTestResult.worldTransform.columns.3.x,
                                       hitTestResult.worldTransform.columns.3.y,
                                       hitTestResult.worldTransform.columns.3.z)
            // 高さを再計算
            updateMeasurement(for: position)
        }
    }

    func updateMeasurement(for position: SCNVector3) {
        // 物体の底と仮定した位置
        let bottomPosition = position

        // 物体の頂点（仮に1.5m上の位置を仮定）
        let topPosition = SCNVector3(bottomPosition.x, bottomPosition.y + 1.5, bottomPosition.z)

        // 高さを計算
        let height = topPosition.y - bottomPosition.y

        // 高さを更新
        DispatchQueue.main.async {
            if height != self.lastHeight {
                self.measurementLabel.text = String(format: "Height: %.2f meters", height)
                self.lastHeight = height
            }
        }
    }
}
*/
