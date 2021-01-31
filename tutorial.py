# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Gitからプールしない場合
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# データセットをダウンロードする
!wget https://sagacentralstorage.blob.core.windows.net/dataset/potato-chips.zip

# zipファイルを解凍する
!unzip potato-chips.zip
# いらないファイルを削除する
!rm -r potato-chips.zip
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Gitからプールする場合
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# レポシトリをクローンする
!git clone https://[username]:[passsword]@github.com/lee4205/Potato_Chip_Classification.git

# メールとユーザ名の設定
!git config --global user.email "[email@gmail.com]"
!git config --global user.name "[username]"

# ワークスペースに移動する
cd [Potato_Chip_Classificationのパス]

# 作業ブランチを作成する
!git branch [ブランチ名]

# 作業ブランチに切り替える
!git checkout [ブランチ名]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




# masterブランチからプールする
!git pull origin master
