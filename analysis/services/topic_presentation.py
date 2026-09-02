"""Presentation-only naming for immutable semantic topic memberships."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from analysis.domain import TopicCluster, TopicRepresentativeComment
from analysis.preprocessing import prepare_analysis_text


TOPIC_NAMING_VERSION = "topic_presentation_v1"
_TOKEN_RE = re.compile(r"[\u3400-\u9fff]{2,}|[A-Za-z][A-Za-z0-9+#.-]{1,}")
_IGNORED = {
    "什么", "这个", "那个", "怎么", "就是", "可以", "还是", "没有", "感觉",
    "真的", "一个", "一下", "不是", "自己", "已经", "还有", "但是", "因为",
    "所以", "然后", "我们", "你们", "他们", "bilibili", "b站", "url", "网址",
    "链接", "视频链接", "链接分享", "回复", "复制", "amp",
}
_CATEGORIES = (
    ("访问与链接问题", ("链接失效", "打不开", "无法访问", "链接没了", "不能看")),
    ("资源与内容获取", ("资源", "链接", "网课", "短剧", "视频", "哪里看", "在哪看", "求分享")),
    ("价格与付费", ("价格", "太贵", "收费", "付费", "多少钱", "便宜")),
    ("咨询与参与方式", ("联系", "报名", "加入", "怎么学", "想学", "咨询")),
    ("兼职与就业", ("兼职", "接单", "工作", "就业", "招聘")),
)


@dataclass(frozen=True)
class TopicPresentation:
    name: str
    summary: str
    naming_version: str = TOPIC_NAMING_VERSION
    provider: str = "deterministic"
    model: str = "representative_ngram_v1"
    prompt_version: str = "none"


def _representative_terms(texts: Sequence[str]) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        seen: set[str] = set()
        for chunk in _TOKEN_RE.findall(prepare_analysis_text(text)):
            if chunk[0].isascii():
                seen.add(chunk.casefold())
                continue
            for size in (4, 3, 2):
                for index in range(max(len(chunk) - size + 1, 0)):
                    token = chunk[index:index + size]
                    if token not in _IGNORED:
                        seen.add(token)
        counts.update(seen)
    ranked = sorted(counts, key=lambda token: (-counts[token], -len(token), token))
    selected: list[str] = []
    for token in ranked:
        if token.casefold() in _IGNORED:
            continue
        if any(token in prior or prior in token for prior in selected):
            continue
        selected.append(token)
        if len(selected) == 2:
            break
    return selected


def present_topic(
    topic: TopicCluster,
    representatives: Sequence[TopicRepresentativeComment],
) -> TopicPresentation:
    """Name a cluster without changing membership or invoking a remote model."""

    exact = [
        prepare_analysis_text(item.value)
        for item in [*topic.top_topics, *topic.top_keywords]
        if prepare_analysis_text(item.value).casefold() not in _IGNORED
    ]
    terms = exact[:2] or _representative_terms([item.text for item in representatives])
    examples = [prepare_analysis_text(item.text) for item in representatives]
    examples = [item for item in examples if item]
    evidence_text = " ".join([*exact, *examples])
    category = next((name for name, cues in _CATEGORIES if any(cue in evidence_text for cue in cues)), None)
    name = category or ("与".join(terms) if terms else "其他讨论")
    summary = f"用户主要讨论{name}" if category else (f"该主题主要围绕{'、'.join(terms)}展开" if terms else "该主题由语义相近的评论组成")
    summary += f"，代表性反馈包括“{examples[0][:48]}”。" if examples else "。"
    return TopicPresentation(name=name, summary=summary)


def present_topic_set(
    items: Sequence[tuple[TopicCluster, Sequence[TopicRepresentativeComment]]],
) -> dict[str, TopicPresentation]:
    """Return stable, unique display names without touching memberships."""

    base = {str(topic.id): present_topic(topic, representatives) for topic, representatives in items}
    groups: dict[str, list[tuple[TopicCluster, Sequence[TopicRepresentativeComment]]]] = {}
    for topic, representatives in items:
        groups.setdefault(base[str(topic.id)].name, []).append((topic, representatives))
    resolved = dict(base)
    for name, group in groups.items():
        if len(group) == 1:
            continue
        used: set[str] = set()
        for topic, representatives in sorted(group, key=lambda item: str(item[0].id)):
            terms = _representative_terms([
                *(item.value for item in [*topic.top_topics, *topic.top_keywords]),
                *(item.text for item in representatives),
            ])
            qualifier = next((term for term in terms if term not in used), "讨论")
            used.add(qualifier)
            original = base[str(topic.id)]
            display_name = f"{name} · {qualifier}"
            resolved[str(topic.id)] = TopicPresentation(
                name=display_name,
                summary=original.summary.replace(name, display_name, 1),
            )
    return resolved
