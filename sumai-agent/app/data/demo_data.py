"""スマイエージェント — デモ用サンプルデータ"""

# ペルソナ A（初めて検討層）のデモシナリオプリセット
DEMO_PERSONA_A = {
    "name": "ペルソナA：初めて検討層",
    "description": "30代夫婦＋未就学児。展示場に行く前に、予算と条件でまず試したい。",
    "preset_message": (
        "はじめまして。家を建てることを検討しています。"
        "夫婦と子供1人（3歳）の3人家族です。"
        "予算は土地込みで4000万円くらいを考えています。"
        "土地はまだ持っていないので、埼玉県内で探す予定です。"
        "リビングが広い家にしたいと思っています。"
    ),
}

DEMO_PERSONA_B = {
    "name": "ペルソナB：建替え・住替え層",
    "description": "50代夫婦。老朽化した自宅を建て替えたい。子供2名が巣立ちつつある。",
    "preset_message": (
        "築30年の自宅を建て替えたいと考えています。"
        "夫婦2人（子供は独立予定）で、将来は1人になる可能性もあります。"
        "建物予算は2500万円程度。土地は既に所有（埼玉・約40坪）。"
        "バリアフリーで管理しやすいコンパクトな家が希望です。"
    ),
}

DEMO_PERSONA_C = {
    "name": "ペルソナC：土地ありこだわり層",
    "description": "40代夫婦＋子2名。相続した土地で自由設計。法規チェックが見せ場。",
    # 法規チェックAI が実データで判定できるよう、敷地面積・用途地域・前面道路幅員を
    # 初回発話に含めている（土地ありのため要確認フラグではなく数値判定が走る）。
    "preset_message": (
        "親から相続した土地（さいたま市、約50坪＝165㎡、第一種低層住居専用地域、"
        "前面道路の幅員は4m）に家を建てたいです。"
        "夫婦と子供2人（小学生）の4人家族。"
        "建物予算は3000〜3500万円。木造2階建てを希望。"
        "在宅勤務があるので書斎が欲しいです。"
    ),
}

# デモ用サンプルデータ一覧
DEMO_PRESETS = [DEMO_PERSONA_A, DEMO_PERSONA_B, DEMO_PERSONA_C]

# ハウスメーカー（デモ用静的データ）
# type: "builder" = 注文住宅系メーカー, "portal" = 不動産情報ポータル
DEMO_MAKERS = [
    # ─── 大手・プレミアム帯 ───────────────────────────
    {
        "name": "積水ハウス",
        "type": "builder",
        "construction_method": "鉄骨・木造",
        "strengths": ["高品質・長期60年保証", "全国展開・施工実績No.1クラス", "ZEH・スマートホーム対応"],
        "price_band": "高め（坪単価80〜120万円）",
        "best_for": ["高品質・長期安心重視", "全国どこでも建てたい"],
        "website": "https://www.sekisuihouse.co.jp/",
    },
    {
        "name": "大和ハウス工業",
        "type": "builder",
        "construction_method": "鉄骨",
        "strengths": ["鉄骨構造の耐震・耐久性", "大空間・間取り自由度が高い", "省エネ・スマートホーム対応"],
        "price_band": "高め（坪単価75〜110万円）",
        "best_for": ["広いLDK・大空間希望", "耐震性を重視"],
        "website": "https://www.daiwahouse.co.jp/",
    },
    {
        "name": "住友林業",
        "type": "builder",
        "construction_method": "木造（BF構法）",
        "strengths": ["木造の高品質・自然素材", "デザイン自由度が高い", "木材の調達力・職人技術"],
        "price_band": "高め（坪単価80〜120万円）",
        "best_for": ["木の温もりあるデザイン希望", "こだわりの自由設計"],
        "website": "https://sfc.jp/",
    },
    {
        "name": "ヘーベルハウス（旭化成）",
        "type": "builder",
        "construction_method": "ALC外壁・鉄骨",
        "strengths": ["高耐震・60年超の耐久性", "都市型3階建て・狭小地に強い", "防火・防音性能"],
        "price_band": "高め（坪単価85〜130万円）",
        "best_for": ["都市部・狭小地での建築", "耐震・耐久性を最優先"],
        "website": "https://www.asahi-kasei.co.jp/hebel/",
    },
    {
        "name": "ミサワホーム",
        "type": "builder",
        "construction_method": "木質パネル工法",
        "strengths": ["蔵・大収納の独自設計", "耐震性の高い木質パネル", "ZEH・省エネ住宅"],
        "price_band": "中〜高め（坪単価70〜100万円）",
        "best_for": ["大容量収納・蔵のある家", "耐震性と収納を両立したい"],
        "website": "https://www.misawa.co.jp/",
    },
    {
        "name": "パナソニック ホームズ",
        "type": "builder",
        "construction_method": "鉄骨",
        "strengths": ["パナソニック系の最新設備連携", "太陽光・蓄電池・スマートホーム", "耐震・耐久性"],
        "price_band": "高め（坪単価80〜115万円）",
        "best_for": ["スマートホーム・省エネ重視", "最新設備を充実させたい"],
        "website": "https://homes.panasonic.com/",
    },
    # ─── ミドル帯 ───────────────────────────
    {
        "name": "住友不動産",
        "type": "builder",
        "construction_method": "木造（2×4・軸組）",
        "strengths": ["都市部の狭小・変形地対応", "デザイン性と品質のバランス", "リフォームとの一体提案"],
        "price_band": "中〜高め（坪単価65〜95万円）",
        "best_for": ["都市部・変形地での建築", "コストとデザインのバランス"],
        "website": "https://www.sumitomo-rd.co.jp/",
    },
    {
        "name": "トヨタホーム",
        "type": "builder",
        "construction_method": "鉄骨ユニット工法",
        "strengths": ["工場生産の高精度・短工期", "耐震性が高いユニット構造", "トヨタグループのアフターサービス"],
        "price_band": "中〜高め（坪単価70〜100万円）",
        "best_for": ["工期を短くしたい", "高精度・工場品質を求める"],
        "website": "https://www.toyotahome.co.jp/",
    },
    # ─── コストパフォーマンス帯 ───────────────────────────
    {
        "name": "タマホーム",
        "type": "builder",
        "construction_method": "木造（軸組工法）",
        "strengths": ["業界最高水準のコスパ", "標準仕様が充実", "全国展開・安心の実績"],
        "price_band": "リーズナブル（坪単価45〜65万円）",
        "best_for": ["予算を抑えて広い家を建てたい", "シンプル・スタンダードな家"],
        "website": "https://www.tamahome.jp/",
    },
    {
        "name": "アイダ設計",
        "type": "builder",
        "construction_method": "木造（軸組工法）",
        "strengths": ["超ローコスト住宅", "シンプル仕様で価格を徹底抑制", "関東圏に強い"],
        "price_band": "ローコスト（坪単価35〜55万円）",
        "best_for": ["とにかく予算最優先", "シンプルな家で十分"],
        "website": "https://www.aidagroup.co.jp/",
    },
    # ─── 不動産情報ポータル（メーカー比較・土地探し） ───────────────────────────
    {
        "name": "SUUMO（スーモ）",
        "type": "portal",
        "construction_method": None,
        "strengths": ["日本最大級の不動産・住宅情報", "複数メーカーを一括比較できる", "土地探し・建売・注文住宅すべてカバー"],
        "price_band": "情報ポータル（メーカー比較・無料）",
        "best_for": ["複数のハウスメーカーを比較したい", "土地と建物をまとめて探したい"],
        "website": "https://suumo.jp/",
    },
    {
        "name": "カナリー",
        "type": "portal",
        "construction_method": None,
        "strengths": ["AI活用の物件マッチング", "賃貸・売買・新築を横断検索", "スマートフォン特化の使いやすさ"],
        "price_band": "情報ポータル（無料）",
        "best_for": ["AIで物件を絞り込みたい", "スマホで手軽に探したい"],
        "website": "https://canary.tools/",
    },
    {
        "name": "LIFULL HOME'S",
        "type": "portal",
        "construction_method": None,
        "strengths": ["全国の注文住宅・建売・土地を網羅", "カタログ一括請求が便利", "住宅展示場・イベント情報が充実"],
        "price_band": "情報ポータル（無料）",
        "best_for": ["まず複数社のカタログを取り寄せたい", "地域のハウスメーカーを探したい"],
        "website": "https://www.homes.co.jp/",
    },
]
