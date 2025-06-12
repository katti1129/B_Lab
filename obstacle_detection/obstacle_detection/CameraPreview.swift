//
//  CameraPreview.swift
//  障害物検知
//
//  Created by かっち on 2024/12/11.
//

import SwiftUI
import AVFoundation

// CameraPreview: ViewControllerをSwiftUIに埋め込む
struct CameraPreview<T: UIViewController>: UIViewControllerRepresentable {
    var viewControllerType: T.Type
    
    func makeUIViewController(context: Context) -> T {
        return T()  // 引数で指定されたUIViewControllerを生成
    }

    func updateUIViewController(_ uiViewController: T, context: Context) {
        // 状態変更時に必要な処理があれば実装
    }
}
