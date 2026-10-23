import time
import warnings
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def lr_kwargs(cfg) -> dict:
    """
        Config -> LogisticRegression kwargs. 
        The l2 penalty is the default of every sklearn
        version, so it is never passed 
        (sklearn >= 1.8 deprecates the `penalty` argument).
    """
    import sklearn
    kw = dict(C=cfg.LR_C, solver=cfg.LR_SOLVER, max_iter=cfg.LR_MAX_ITER, random_state=cfg.SEED)
    
    return kw


@dataclass
class LRBaseline:
    model: object
    cfg: object
    logger: object
    diagnostics: dict
    accuracy: float

    # predictions
    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def decision_function(self, X) -> np.ndarray:
        return self.model.decision_function(X)

    # maps
    def local_map(self, x_flat, class_idx: int) -> np.ndarray:
        """
            Contribution of each pixel to the logit of `class_idx`: (IMAGE_SIZE, IMAGE_SIZE).
        """
        x = np.asarray(x_flat, dtype=np.float64).reshape(-1)
        side = self.cfg.IMAGE_SIZE
        return (self.model.coef_[class_idx] * x).reshape(side, side)

    def global_map(self, class_idx: int) -> np.ndarray:
        """
            Relative coefficients of `class_idx`: (IMAGE_SIZE, IMAGE_SIZE).
        """
        coef = self.model.coef_
        side = self.cfg.IMAGE_SIZE
        return (coef[class_idx] - coef.mean(axis=0)).reshape(side, side)

    # remove this version for test version
    def check_exactness(self, X_sample) -> dict:
        """
            Falsifiable facts about the model, measured on a sample of inputs.
        """
        manual = X_sample @ self.model.coef_.T + self.model.intercept_
        return {
            "decision_exact_max_err": float(np.abs(manual - self.decision_function(X_sample)).max()),
            "coef_colsum_max": float(np.abs(self.model.coef_.sum(axis=0)).max()),
            "classes_ok": bool(np.array_equal(self.model.classes_, np.arange(self.cfg.NUM_CLASSES))),
        }
    def score_baseline(self,y_t,X_t , cfg,logger):
        """
            print result and score about Logistic Regression.
        """


        self.accuracy = accuracy_score(y_t, self.predict(X_t))
        logger.info(f" test Accuracy (Logistic Regression): {self.accuracy}")
        
        tag_diag = "LogisticRegression"
        low, high = cfg.THRESHOLDS["lr_expected_acc_range"]

        logger.info( f"{tag_diag}: fit {self.diagnostics['fit_seconds']:.1f}s, "
                     f"n_iter {self.diagnostics['n_iter']}, "
                     f"val acc {self.diagnostics['val_accuracy']:.4f}"
                    )

        # Convergence
        if self.diagnostics["convergence_warning"]:
            logger.info( f"[WARN] {tag_diag}: ConvergenceWarning=True -> "
                     f"consider raising LR_MAX_ITER (now {cfg.LR_MAX_ITER})"
                    )
        else:
            logger.info(f"[OK] {tag_diag}: ConvergenceWarning=False")

        # Validation accuracy
        acc = self.diagnostics["val_accuracy"]
        if low <= acc <= high:
            logger.info( f"[OK] {tag_diag}: validation accuracy {acc:.4f} "
                         f"(expected ~0.92, range {lo}-{hi})"
                       )
        else:
            logger.info(f"[WARN] {tag_diag}: validation accuracy {acc:.4f} "
                        f"(expected ~0.92, range {lo}-{hi})"
                       )
        # Classi
        if self.diagnostics["classes_ok"]:
            logger.info(f"[OK] {tag}: classes_ == 0..9 (coefficient rows aligned)")
        else:
            logger.info(f"[FAIL] {tag_diag}: classes_ are not aligned with 0..9")

        # Altri warning
        for warning in self.diagnostics["other_warnings"]:
            logger.info(f"[INFO] {tag_diag}: warning raised -> {warning}")

def fit_baseline(X_t, y_t, X_v , y_v, cfg , logger ) -> LRBaseline:
    """
         Fit on train A ConvergenceWarning is re-emitted, never swallowed.
    """
    
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.time()
        model = LogisticRegression(**lr_kwargs(cfg)).fit(X[idx_train], y[idx_train])
        fit_seconds = time.time() - t0
   
    categories = sorted({w.category.__name__ for w in caught})
    diagnostics = {
        "fit_seconds": fit_seconds,
        "n_iter": int(np.max(model.n_iter_)),
        "convergence_warning": any(w.category.__name__ == "ConvergenceWarning" for w in caught),
        "other_warnings": sorted({f"{w.category.__name__}: {str(w.message)[:140]}"
                                  for w in caught if w.category.__name__ != "ConvergenceWarning"}),
        "n_train": int(len(len(X_t))),
        "val_accuracy": float(model.score(X_v, y_v)),
        "classes_ok": bool(np.array_equal(model.classes_, np.arange(cfg.NUM_CLASSES))),

    }
    return LRBaseline(model=model, cfg=cfg,logger=logger, diagnostics=diagnostics)
