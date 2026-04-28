"""
modeling/ensemble.py — Component responsible for constructing ensemble models, specifically Voting Classifiers.
"""
from __future__ import annotations

from sklearn.ensemble import VotingClassifier

from src.modeling.metrics import agregar_metricas_folds


class EnsembleBuilder:
    def build_voting(self, top_n_entries, X_train, y_train):
        melhor_voting = VotingClassifier(
            estimators=self._construir_estimadores_base(top_n_entries),
            voting='soft'
        )
        fold_mets = self.cv_runner.executar(melhor_voting, X_train, y_train)
        return {**agregar_metricas_folds(fold_mets), "_instance": melhor_voting}
