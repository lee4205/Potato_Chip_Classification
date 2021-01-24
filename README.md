# 作業チュートリアル

GoogleColabを新規で開くとき

* レポシトリをクローンする（ファイルが入ったら、スキップ）

```!git clone https://[username]:[passsword]@github.com/lee4205/Potato_Chip_Classification.git```

* ユーザ名とメールの設定

```!git config --global user.email "[email@gmail.com]"```

```!git config --global user.name "[username]"```

* ワークスペースに移動する

```cd [Potato_Chip_Classificationのパス]```

* 最新の変更をプール

```!git pull origin master```

* ブランチの作成（ブランチができたら、スキップ)

```!git branch [ブランチ名]```

* ブランチの切り替え

```!git checkout [ブランチ名]```

～～～～～～～～～～～～～～～～～～～～～～～～～～～作業始める～～～～～～～～～～～～～～～～～～～～～～～～～～～～～

～～～～～～～～～～～～～～～～～～～～～～～～～～～作業終わる～～～～～～～～～～～～～～～～～～～～～～～～～～～～～

* ファイル変更を追加する

```!git add [ファイル名]```

* commitのメッセージを書く

```!git commit -m "[commitのメッセージ]"```

* ファイル変更をgitにプッシュする

```!git push origin [ブランチ名]```

～～～～～～～～～～～～～～～～～～～～～～～～～～～役に立つコマンド～～～～～～～～～～～～～～～～～～～～～～～～～～

* ファイル変更のみ取り消し（add、commitは取り消せない）

```!git checkout .```

* commit、addとファイル変更の取り消し

```!git reset --hard```

* commitとaddの取り消し

```!git reset --mixed```

* commitのみ取り消し

```!git reset --soft```

* 変更の状態を確認する

```!git status```

* ログを見る

```!git log```

* ワークスペースにある全部のファイルをリストする

```ls```
