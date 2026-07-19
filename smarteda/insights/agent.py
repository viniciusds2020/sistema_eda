"""Deterministic insights and optional Groq/Llama interpretation."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from smarteda.insights.context import InsightContextBuilder


class InsightAgentError(RuntimeError):
    """Raised when the external insight provider cannot answer safely."""


Transport = Callable[[str, dict[str, str], bytes, float], dict[str, Any]]


class InsightAgent:
    """Answer questions from compact SmartEDA statistics, never dataset rows."""

    def __init__(
        self,
        provider: str = "rules",
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        max_context_items: int = 30,
        timeout: float = 30.0,
        transport: Optional[Transport] = None,
    ):
        if provider not in {"rules", "groq"}:
            raise ValueError("provider must be 'rules' or 'groq'.")
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.timeout = timeout
        self.context_builder = InsightContextBuilder(max_items=max_context_items)
        self.transport = transport or self._http_transport

    def summarize(self, results: dict[str, Any]) -> dict[str, Any]:
        context = self.context_builder.build(results)
        deterministic = self._rule_summary(context)
        if self.provider == "rules":
            return deterministic
        prompt = (
            "Gere JSON válido com as chaves executive_summary, findings e recommendations. "
            "Use somente as evidências fornecidas, cite métricas, não invente causalidade e "
            "mantenha cada lista com no máximo 5 itens.\nCONTEXT="
            + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        )
        return self._groq_json(prompt)

    def ask(self, question: str, results: dict[str, Any]) -> str:
        question = " ".join(question.split())[:500]
        if not question:
            raise ValueError("question must not be empty.")
        context = self.context_builder.build(results)
        if self.provider == "rules":
            summary = self._rule_summary(context)
            return self._rule_answer(question, summary)
        prompt = (
            "Responda em português usando exclusivamente o contexto estatístico. "
            "Diferencie evidência, hipótese e recomendação. Se não houver evidência, diga isso. "
            f"Pergunta: {question}\nCONTEXT="
            + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        )
        return self._groq_text(prompt)

    def _rule_summary(self, context: dict[str, Any]) -> dict[str, Any]:
        overview = context.get("overview", {})
        quality = context.get("quality_diagnostics", {})
        drift = context.get("train_test_profile", {})
        tests = context.get("statistical_drift_tests", {})
        conditioned = context.get("target_conditioned_drift", {})
        longitudinal = context.get("longitudinal_monitoring", {})
        preprocessing = context.get("preprocessing_diagnostics", {})
        findings: list[str] = []
        recommendations: list[str] = []

        missing = float(overview.get("total_missing_pct", 0) or 0)
        if missing > 0.1:
            findings.append(f"Ausência global elevada: {missing:.1%}.")
            recommendations.append("Investigar mecanismo de ausência antes da imputação.")
        qsummary = quality.get("summary", {})
        ids = int(qsummary.get("possible_id_columns", 0) or 0)
        leakage = int(qsummary.get("possible_leakage_warnings", 0) or 0)
        constants = len(quality.get("constant_columns", []))
        if ids:
            findings.append(f"Foram detectadas {ids} possíveis colunas de ID.")
            recommendations.append("Excluir possíveis IDs da modelagem após validação de domínio.")
        if leakage:
            findings.append(f"Existem {leakage} alertas de possível target leakage.")
            recommendations.append("Bloquear promoção do modelo até revisar o possível leakage.")
        if constants:
            findings.append(f"Existem {constants} variáveis constantes.")
            recommendations.append("Remover variáveis constantes do pipeline de features.")
        dsummary = drift.get("summary", {})
        high_drift = int(dsummary.get("high_drift_features", 0) or 0)
        if high_drift:
            findings.append(f"Há {high_drift} variáveis com drift alto entre treino e teste.")
            recommendations.append("Priorizar variáveis com maior efeito e impacto no modelo.")
        significant = int(tests.get("summary", {}).get("significant_after_correction", 0) or 0)
        if significant:
            findings.append(f"{significant} testes permanecem significativos após correção.")
        conditional = int(conditioned.get("summary", {}).get("high_drift_findings", 0) or 0)
        if conditional:
            findings.append(f"Drift alto condicionado ao target em {conditional} achados.")
        windows = int(longitudinal.get("summary", {}).get("windows_with_high_drift", 0) or 0)
        if windows:
            findings.append(f"{windows} janelas longitudinais apresentam drift alto.")
        prep_summary = preprocessing.get("summary", {})
        prep_actions = preprocessing.get("prioritized_actions", [])
        if prep_actions:
            findings.append(
                f"O diagnóstico de pré-processamento priorizou {len(prep_actions)} ações."
            )
            recommendations.extend(
                str(row.get("action", "")) for row in prep_actions[:3] if row.get("action")
            )
        non_normal = int(prep_summary.get("non_normal_numeric_columns", 0) or 0)
        if non_normal:
            findings.append(f"{non_normal} variáveis numéricas rejeitaram normalidade.")
        if not findings:
            findings.append("Nenhum alerta crítico foi encontrado nos resultados disponíveis.")
            recommendations.append(
                "Manter monitoramento e validar os achados com conhecimento de domínio."
            )
        return {
            "provider": self.provider,
            "model": self.model if self.provider == "groq" else None,
            "executive_summary": " ".join(findings[:3]),
            "findings": findings[:8],
            "recommendations": list(dict.fromkeys(recommendations))[:8],
            "privacy": "Somente estatísticas agregadas; nenhuma linha do dataset foi enviada.",
        }

    @staticmethod
    def _rule_answer(question: str, summary: dict[str, Any]) -> str:
        normalized = question.casefold()
        if any(term in normalized for term in ("recom", "fazer", "ação", "acao")):
            values = summary["recommendations"]
        elif any(term in normalized for term in ("resumo", "executivo", "geral")):
            return summary["executive_summary"]
        else:
            values = summary["findings"]
        return " ".join(f"{index + 1}. {item}" for index, item in enumerate(values))

    def _groq_json(self, prompt: str) -> dict[str, Any]:
        text = self._groq_text(prompt, response_format={"type": "json_object"})
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InsightAgentError("Groq returned invalid JSON.") from exc
        parsed["provider"] = "groq"
        parsed["model"] = self.model
        parsed["privacy"] = "Somente estatísticas agregadas; nenhuma linha do dataset foi enviada."
        return parsed

    def _groq_text(self, prompt: str, response_format: Optional[dict[str, str]] = None) -> str:
        if not self.api_key:
            raise InsightAgentError("Set GROQ_API_KEY to use provider='groq'.")
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800,
            "messages": [
                {"role": "system", "content": "Você é um analista estatístico cuidadoso."},
                {"role": "user", "content": prompt},
            ],
        }
        if response_format:
            payload["response_format"] = response_format
        response = self.transport(
            "https://api.groq.com/openai/v1/chat/completions",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json.dumps(payload).encode("utf-8"),
            self.timeout,
        )
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise InsightAgentError("Unexpected Groq response.") from exc

    @staticmethod
    def _http_transport(
        url: str, headers: dict[str, str], data: bytes, timeout: float
    ) -> dict[str, Any]:
        try:
            with urlopen(
                Request(url, data=data, headers=headers, method="POST"), timeout=timeout
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise InsightAgentError(f"Groq request failed: {exc}") from exc
