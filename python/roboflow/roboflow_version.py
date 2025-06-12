import roboflow

# Roboflow APIキーを設定
rf = roboflow.Roboflow(api_key="x779kszAVC0k6cpKTDJy")

# プロジェクトを取得
project = rf.workspace().project("stairs-3kevq")

# プロジェクト内の全てのバージョンを取得
versions = project.versions()

# 各バージョンの情報を表示
for version in versions:
    print(f"Version ID: {version.id}, Name: {version.name}, Model Type: {version.model_type}")
