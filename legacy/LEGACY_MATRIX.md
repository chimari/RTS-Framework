| Legacy Step    | 実際の役割           | 入力                   | 出力                     | 現framework対応 |
| -------------- | --------------- | -------------------- | ---------------------- | ------------ |
| 01             | 入力検証・manifest固定 | temperature index    | frame manifest         | ほぼ対応         |
| 02             | 全期間raw統計・診断     | frame manifest + raw | 統計map                  | 現在欠落         |
| 03             | 温度bin別粗候補抽出     | frame manifest + raw | candidate catalog/mask | 現在欠落         |
| direct 2-state | legacy外の代替探索    | manifest/stack       | direct candidates      | 現Step04付近    |
| 04 | 候補時系列抽出 | current Step03/04のstack処理 | 再利用可能 | 独立Step化 |
| 05 | dataset別中心化 | 対応不明 | 欠落疑い | 再実装 |
| 06 | histogram state検出 | current Step04の一部 | 再配置 | legacy比較 |
| 07 | state assignment | current Step04/05の一部 | 混在 | 分離 |
| 08 | transition統計 | current Step04の一部 | 混在 | 分離 |
| 09 | 分類・辞書生成 | current Step04とは別物 | 欠落 | 再構築 |
