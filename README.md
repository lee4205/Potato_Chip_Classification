# 作業チュートリアル

GoogleColabを新規で開くとき

-レポシトリをクローンする
```!git clone https://[username]:[passsword]@github.com/lee4205/Potato_Chip_Classification.git```

-ユーザ名とメールの設定
```!git config --global user.email "[email@gmail.com]"```
```!git config --global user.name "[username]"```
-ワークスペースに移動する
cd [Potato_Chip_Classificationのパス]
-最新の変更をプール
!git pull origin master
-ブランチの作成（できたら、スキップ)
!git branch [ブランチ名]
-ブランチの切り替え
!git checkout [ブランチ名]

～～～～～～～～～～～～～～～～～～～～～～～～～～～作業を始める～～～～～～～～～～～～～～～～～～～～～～～～～～～～

作業終わったら
-ファイルを追加
!git add [ファイル名]
-commitのメッセージを書く
!git commit -m "[commitのメッセージ]"
-変更をgitにプッシュする
!git push origin [ブランチ名]

～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～

役に立つコマンド
-変更を行う前の状態に戻す
!git checkout .
-全部に戻す
!git reset --hard
-commitとaddの取り消し
!git reset --mixed
-commitのみ取り消し
!git reset --soft
-変更の状態を確認する
!git status
-ログを見る
!git log
-ワークスペースにある全部のファイルをリストする
ls
