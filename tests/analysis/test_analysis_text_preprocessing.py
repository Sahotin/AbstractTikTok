from analysis.preprocessing import prepare_analysis_text


def test_analysis_text_removes_url_and_query_garbage() -> None:
    text = "链接失效 https://www.bilibili.com/video/BV1xx?spm_id_from=333&amp;share_source=copy"
    assert prepare_analysis_text(text) == "链接失效"


def test_analysis_text_decodes_entities_and_removes_markup() -> None:
    assert prepare_analysis_text("价格&amp;质量 &gt; 宣传<br>确实离谱") == "价格&质量 > 宣传 确实离谱"


def test_analysis_text_removes_reply_prefix_and_mentions() -> None:
    assert prepare_analysis_text("回复 @小明：这个链接已经失效") == "这个链接已经失效"
    assert prepare_analysis_text("@小明 这个价格太贵了") == "这个价格太贵了"


def test_analysis_text_preserves_meaningful_emoji_comment() -> None:
    assert prepare_analysis_text("哈哈哈哈哈哈这个价格太离谱了😂") == "哈哈哈这个价格太离谱了😂"
    assert prepare_analysis_text("😂😂😂") == "😂😂😂"


def test_analysis_text_removes_uuid_and_meaningless_numbers() -> None:
    assert prepare_analysis_text("任务 392876f1-1234-4abc-8def-123456789abc 失败") == "任务 失败"
    assert prepare_analysis_text("123456789") == ""
    assert prepare_analysis_text("1") == ""


def test_analysis_text_keeps_normal_chinese_comment_and_raw_value() -> None:
    raw = "  课程讲得很好，适合新手。  "
    assert prepare_analysis_text(raw) == "课程讲得很好,适合新手"
    assert raw == "  课程讲得很好，适合新手。  "
