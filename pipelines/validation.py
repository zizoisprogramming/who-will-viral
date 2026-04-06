import ast
import re
from datetime import datetime
from itertools import groupby

import great_expectations as gx
import numpy as np
import pandas as pd


def quick_summary(df: pd.DataFrame):
    """Fast dataset snapshot printed before any validation runs."""
    print("\n" + "=" * 65)
    print("  DATASET QUICK SUMMARY")
    print("=" * 65)
    print(f"  Rows    : {len(df)}")
    print(f"  Columns : {df.shape[1]}")

    if "is_trending" in df.columns:
        counts = df["is_trending"].value_counts(dropna=False)
        pct    = df["is_trending"].value_counts(normalize=True, dropna=False) * 100
        print("\n  Trending Distribution:")
        for val in counts.index:
            print(f"    {val} → {counts[val]} rows ({pct[val]:.1f}%)")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\n  Top 10 Missing Values:")
    if not missing.empty:
        for col, cnt in missing.head(10).items():
            print(f"    {col:<45} {cnt}")
    else:
        print("    None ✅")

    print("\n  Numeric Summary (mean / min / max):")
    num = df.select_dtypes(include=np.number).describe().T
    for col in num.index[:8]:
        print(f"    {col:<35}  mean={num.loc[col, 'mean']:>12.1f}  "
              f"min={num.loc[col, 'min']:>10.0f}  "
              f"max={num.loc[col, 'max']:>12.0f}")
    print("=" * 65)


def run_gx_validation(df: pd.DataFrame):
    """
    GX owns every check that can be expressed as a declarative expectation:
      - not-null               → Completeness
      - value >= 0 / regex     → Accuracy
      - compound uniqueness    → Uniqueness
      - column set / dtypes    → Consistency
      - categorical sets       → Consistency
      - row count / median     → Distribution
      - value length           → Structural
    """
    context     = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(name="pandas_source")
    data_asset  = data_source.add_dataframe_asset(name="videos_asset")
    batch_def   = data_asset.add_batch_definition_whole_dataframe("batch")
    suite       = context.suites.add(gx.ExpectationSuite(name="videos_suite"))

    # ── COMPLETENESS: required columns must not be null ──────────
    for col in ["video_id", "title", "view_count", "likes", "publishedAt", "channelId"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )

    # ── ACCURACY: numeric lower bounds ──────────────────────────
    for col in ["view_count", "likes", "comment_count", "favoriteCount"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(column=col, min_value=0)
        )

    # ── ACCURACY: duration must be valid ISO-8601 ────────────────
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToMatchRegex(
            column="duration",
            regex=r"^P(?:(\d+)D)?T(?=\d+[HMS])(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$"
        )
    )

    # ── ACCURACY: publishedAt must start with YYYY-MM-DD ─────────
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToMatchRegex(
            column="publishedAt",
            regex=r"^\d{4}-\d{2}-\d{2}"
        )
    )

    # ── UNIQUENESS: composite key ────────────────────────────────
    suite.add_expectation(
        gx.expectations.ExpectCompoundColumnsToBeUnique(
            column_list=list(df.columns)
        )
    )

    # ── CONSISTENCY: exact column set ───────────────────────────
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=[
                "video_id", "title", "publishedAt", "channelId", "channelTitle",
                "categoryId", "trending_date", "tags", "view_count", "likes",
                "comment_count", "thumbnail_link", "description", "is_trending",
                "defaultLanguage", "duration", "dimension", "definition", "caption",
                "licensedContent", "projection", "embeddable", "madeForKids",
                "favoriteCount", "contentDetails.regionRestriction.blocked",
                "contentDetails.regionRestriction.allowed",
                "contentDetails.contentRating.ytRating", "chapter_count",
                "chapters", "playability_status", "supports_miniplayer",
                "card_count", "cards", "is_verified", "badge_labels",
                "comments_disabled", "has_paid_promotion"
            ]
        )
    )

    # ── CONSISTENCY: categorical allowed values ──────────────────
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="defaultLanguage",
            value_set=[
                "en", "en-US", "en-GB", "en-IN",
                "ar", "ar-EG", "ar-SA",
                "es", "es-ES", "es-419",
                "fr", "fr-FR", "de", "it",
                "pt", "pt-BR", "ru", "hi", "id",
                "ja", "ko", "zh", "zh-CN", "zh-TW",
                "tr", "nl", "pl", "sv", "th", "vi"
            ]
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="dimension", value_set=["2d", "3d"]
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="definition", value_set=["hd", "sd"]
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="projection", value_set=["rectangular", "360"]
        )
    )

    # ── DISTRIBUTION: row count bounds ───────────────────────────
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=100, max_value=500_000)
    )

    # ── DISTRIBUTION: view_count median > 0 ─────────────────────
    suite.add_expectation(
        gx.expectations.ExpectColumnMedianToBeBetween(column="view_count", min_value=1)
    )

    # ── STRUCTURAL: video_id must be exactly 11 chars ────────────
    suite.add_expectation(
        gx.expectations.ExpectColumnValueLengthsToBeBetween(
            column="video_id", min_value=11, max_value=11
        )
    )

    # ── RUN ──────────────────────────────────────────────────────
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(name="videos_validation", data=batch_def, suite=suite)
    )
    results = validation_def.run(batch_parameters={"dataframe": df})
    _print_gx_report(results)
    return results


def _print_gx_report(results):
    print("\n" + "=" * 65)
    print("  GREAT EXPECTATIONS REPORT  (v1.x)")
    print("=" * 65)
    print(f"  Overall : {'PASSED' if results.success else 'FAILED'}")
    print("=" * 65)

    for exp in results.results:
        exp_type = exp.expectation_config.type
        col      = exp.expectation_config.kwargs.get("column", "table-level")
        status   = "PASS" if exp.success else "FAIL"
        print(f"\n  [{status}]  {exp_type}")
        print(f"             Column : {col}")
        if not exp.success and exp.result:
            r = exp.result
            if r.get("unexpected_count"):
                print(f"             Issues : {r['unexpected_count']} unexpected values")
            if r.get("partial_unexpected_list"):
                print(f"             Sample : {r['partial_unexpected_list'][:3]}")

    print("\n" + "=" * 65)


# ═══════════════════════════════════════════════════════════════════
#  3.  PANDAS VALIDATOR  (only what GX cannot express)
# ═══════════════════════════════════════════════════════════════════

class DataValidator:
    """
    Pandas-only checks for logic that falls outside GX's declarative model:
      Completeness  - blank strings
      Accuracy      - cross-column business rules
      Timeliness    - future dates, date order, dataset freshness
      Outliers      - IQR extreme outliers, z-score extremes
      Distribution  - category dominance, zero variance, trending ratio
      Referential   - count_col vs actual len(list_col)
    """

    def __init__(self):
        self.validation_results = []

    def _make_report(self, check_type: str, dimension: str):
        return {
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dimension":  dimension,
            "check_type": check_type,
            "passed":     True,
            "issues":     [],
        }

    def _save(self, report):
        self.validation_results.append(report)
        return report

    # ── COMPLETENESS ────────────────────────────────────────────
    # GX covers not-null; pandas covers blank strings which pass not-null

    def validate_no_blank_strings(self, df, text_columns):
        """Empty-string cells that pass NOT NULL — GX cannot catch these."""
        report = self._make_report("Blank Strings", "Completeness")
        for col in text_columns:
            if col not in df.columns:
                continue
            n = (df[col].astype(str).str.strip() == "").sum()
            if n:
                report["passed"] = False
                report["issues"].append(f"Column '{col}': {n} blank-string values")
        return self._save(report)

    # ── ACCURACY ────────────────────────────────────────────────
    # GX covers per-column ranges & regex; pandas covers cross-column logic

    def validate_cross_column_rules(self, df):
        """
        Business rules that span multiple columns — GX has no multi-column
        conditional expectation for these patterns.
        """
        report = self._make_report("Cross-Column Rules", "Accuracy")

        # is_trending=1 -> trending_date must not be null
        if {"is_trending", "trending_date"}.issubset(df.columns):
            n = df[(df["is_trending"] == 1) & df["trending_date"].isna()].shape[0]
            if n:
                report["passed"] = False
                report["issues"].append(
                    f"is_trending=1 but trending_date is null: {n} rows"
                )

        # comments_disabled=True -> comment_count must be 0
        if {"comments_disabled", "comment_count"}.issubset(df.columns):
            n = df[
                (df["comments_disabled"] == True) & (df["comment_count"] != 0)
            ].shape[0]
            if n:
                report["passed"] = False
                report["issues"].append(
                    f"comments_disabled=True but comment_count!=0: {n} rows"
                )

        # likes must not exceed view_count
        if {"likes", "view_count"}.issubset(df.columns):
            n = df[df["likes"] > df["view_count"]].shape[0]
            if n:
                report["passed"] = False
                report["issues"].append(
                    f"likes > view_count (impossible): {n} rows"
                )

        # age-restricted videos should have comments disabled
        if {"is_age_restricted", "comments_disabled"}.issubset(df.columns):
            n = df[
                (df["is_age_restricted"] == True) & (df["comments_disabled"] == False)
            ].shape[0]
            if n:
                report["passed"] = False
                report["issues"].append(
                    f"is_age_restricted=True but comments_disabled=False: {n} rows"
                )

        return self._save(report)

    # ── TIMELINESS ──────────────────────────────────────────────
    # GX has no built-in "must be in the past" or "col A <= col B" expectation

    def validate_no_future_dates(self, df, date_columns):
        """GX cannot compare a column value to the current timestamp."""
        report = self._make_report("No Future Dates", "Timeliness")
        now = pd.Timestamp.now(tz="UTC")
        for col in date_columns:
            if col not in df.columns:
                continue
            dates = pd.to_datetime(df[col], errors="coerce", utc=True)
            n = (dates > now).sum()
            if n:
                report["passed"] = False
                report["issues"].append(f"Column '{col}': {n} future timestamps")
        return self._save(report)

    def validate_date_order(self, df, earlier_col, later_col):
        """GX cannot enforce col A <= col B across two date columns."""
        report = self._make_report(
            f"Date Order ({earlier_col} <= {later_col})", "Timeliness"
        )
        if earlier_col not in df.columns or later_col not in df.columns:
            report["issues"].append(
                f"Column(s) not found: '{earlier_col}', '{later_col}'"
            )
            return self._save(report)

        t1 = pd.to_datetime(df[earlier_col], errors="coerce", utc=True)
        t2 = pd.to_datetime(df[later_col],   errors="coerce", utc=True)
        n  = (t1.notna() & t2.notna() & (t2 < t1)).sum()
        if n:
            report["passed"] = False
            report["issues"].append(
                f"{n} rows where '{later_col}' is earlier than '{earlier_col}'"
            )
        return self._save(report)

    def validate_data_freshness(self, df, date_col, max_days_old=365):
        """GX cannot compute how many days ago the latest record was created."""
        report = self._make_report(
            f"Data Freshness (<= {max_days_old} days)", "Timeliness"
        )
        if date_col not in df.columns:
            report["issues"].append(f"Column '{date_col}' not found")
            return self._save(report)

        latest   = pd.to_datetime(df[date_col], errors="coerce", utc=True).max()
        now      = pd.Timestamp.now(tz="UTC")

        if pd.isna(latest):
            report["passed"] = False
            report["issues"].append(f"No valid dates in '{date_col}'")
        else:
            age_days = (now - latest).days
            if age_days > max_days_old:
                report["passed"] = False
            report["issues"].append(
                f"Latest record in '{date_col}' is {age_days} days old "
                f"(threshold: {max_days_old})"
            )
        return self._save(report)

    # ── OUTLIERS ────────────────────────────────────────────────
    # GX has no IQR or z-score expectation

    def validate_outliers_iqr(self, df, numeric_columns, multiplier=5.0):
        """GX has no IQR-based outlier expectation."""
        report = self._make_report(f"Outliers - IQR x{multiplier}", "Outliers")
        for col in numeric_columns:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
            n = ((df[col] < lo) | (df[col] > hi)).sum()
            if n:
                report["passed"] = False
                report["issues"].append(
                    f"Column '{col}': {n} extreme outliers "
                    f"(bounds [{lo:.0f}, {hi:.0f}])"
                )
        return self._save(report)

    def validate_outliers_zscore(self, df, numeric_columns, threshold=5.0):
        """GX has no z-score outlier expectation."""
        report = self._make_report(f"Outliers - Z-score > {threshold}", "Outliers")
        for col in numeric_columns:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if s.std() == 0:
                continue
            n = ((df[col] - s.mean()).abs() / s.std() > threshold).sum()
            if n:
                report["passed"] = False
                report["issues"].append(
                    f"Column '{col}': {n} values with |z| > {threshold}"
                )
        return self._save(report)

    # ── DISTRIBUTION ────────────────────────────────────────────
    # GX covers row count and median; pandas covers ratio/dominance/variance

    def validate_category_dominance(self, df, col, max_share=0.80):
        """GX has no 'top category must not dominate' expectation."""
        report = self._make_report(f"Category Dominance '{col}'", "Distribution")
        if col not in df.columns:
            report["issues"].append(f"Column '{col}' not found")
            return self._save(report)

        top_share = df[col].value_counts(normalize=True).iloc[0]
        if top_share > max_share:
            report["passed"] = False
            report["issues"].append(
                f"Top value covers {top_share:.1%} (max allowed: {max_share:.0%})"
            )
        else:
            report["issues"].append(f"Top value share: {top_share:.1%}")
        return self._save(report)

    def validate_non_zero_variance(self, df, numeric_columns):
        """GX has no zero-variance / constant-column expectation."""
        report = self._make_report("Non-Zero Variance", "Distribution")
        for col in numeric_columns:
            if col not in df.columns:
                continue
            if df[col].std() == 0:
                report["passed"] = False
                report["issues"].append(
                    f"Column '{col}': all values are identical (zero variance)"
                )
        return self._save(report)

    def validate_trending_ratio(self, df, col="is_trending",
                                min_ratio=0.01, max_ratio=0.90):
        """GX's mean expectation is close but ratio bounds are dataset-specific."""
        report = self._make_report("Trending Ratio", "Distribution")
        if col not in df.columns:
            report["issues"].append(f"Column '{col}' not found")
            return self._save(report)

        ratio = df[col].mean()
        if not (min_ratio <= ratio <= max_ratio):
            report["passed"] = False
            report["issues"].append(
                f"Trending ratio {ratio:.1%} outside [{min_ratio:.0%}, {max_ratio:.0%}]"
            )
        else:
            report["issues"].append(f"Trending ratio: {ratio:.1%}")
        return self._save(report)

    # ── REFERENTIAL INTEGRITY ────────────────────────────────────
    # GX has no expectation for count_col == len(parse(list_col))

    def validate_count_matches_list(self, df, count_col, list_col):
        """GX cannot parse a stringified list and compare its length to another column."""
        report = self._make_report(
            f"Count Match: '{count_col}' vs len('{list_col}')",
            "Referential Integrity"
        )
        if count_col not in df.columns or list_col not in df.columns:
            report["issues"].append(
                f"Column(s) not found: '{count_col}', '{list_col}'"
            )
            return self._save(report)

        def _len(val):
            if pd.isna(val) or str(val).strip() in ("", "[]", "nan"):
                return 0
            try:
                return len(ast.literal_eval(str(val)))
            except Exception:
                return -1

        actual   = df[list_col].apply(_len)
        mismatch = (
            actual.ge(0) &
            (df[count_col].fillna(0).astype(int) != actual)
        ).sum()

        if mismatch:
            report["passed"] = False
            report["issues"].append(
                f"{mismatch} rows where '{count_col}' != len('{list_col}')"
            )
        return self._save(report)

    # ── REPORT ───────────────────────────────────────────────────

    def generate_report(self):
        total        = len(self.validation_results)
        passed       = sum(1 for r in self.validation_results if r["passed"])
        failed       = total - passed
        success_rate = (passed / total * 100) if total else 0

        print("\n" + "=" * 65)
        print("  PANDAS VALIDATION REPORT  (GX-exclusive checks excluded)")
        print("=" * 65)
        print(f"  Total Checks  : {total}")
        print(f"  Passed        : {passed}")
        print(f"  Failed        : {failed}")
        print(f"  Success Rate  : {success_rate:.1f}%")
        print("=" * 65)

        sorted_r = sorted(self.validation_results, key=lambda r: r["dimension"])
        print("\n  Per-Dimension Breakdown:")
        for dim, group in groupby(sorted_r, key=lambda r: r["dimension"]):
            items = list(group)
            p, t  = sum(r["passed"] for r in items), len(items)
            bar   = "X" * p + "." * (t - p)
            print(f"    {dim:<25} {bar}  {p}/{t}")

        print()
        for result in self.validation_results:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"\n[{status}]  [{result['dimension']}]  {result['check_type']}")
            print(f"   Time: {result['timestamp']}")
            for issue in result["issues"]:
                print(f"   -> {issue}")
            if not result["issues"]:
                print("   -> No issues found.")

        print("\n" + "=" * 65)

        return {
            "total": total, "passed": passed,
            "failed": failed, "success_rate": success_rate,
            "details": self.validation_results,
        }


# ═══════════════════════════════════════════════════════════════════
#  4.  UNIFIED FINAL SCORECARD
# ═══════════════════════════════════════════════════════════════════

def summarize_all(gx_results, pandas_summary: dict):
    total_gx   = len(gx_results.results)
    passed_gx  = sum(r.success for r in gx_results.results)
    total_pd   = pandas_summary["total"]
    passed_pd  = pandas_summary["passed"]
    total_all  = total_gx + total_pd
    passed_all = passed_gx + passed_pd
    pct        = passed_all / total_all * 100 if total_all else 0

    print("\n" + "=" * 65)
    print("  FINAL VALIDATION SCORECARD")
    print("=" * 65)
    print(f"  Great Expectations  :  {passed_gx:>3} / {total_gx}  passed")
    print(f"  Pandas Checks       :  {passed_pd:>3} / {total_pd}  passed")
    print(f"  {'-' * 42}")
    print(f"  TOTAL               :  {passed_all:>3} / {total_all}  passed  ({pct:.1f}%)")
    print("=" * 65)

    details  = pandas_summary["details"]
    sorted_r = sorted(details, key=lambda r: r["dimension"])
    print("\n  Per-Dimension Breakdown (pandas checks):")
    for dim, group in groupby(sorted_r, key=lambda r: r["dimension"]):
        items = list(group)
        p, t  = sum(r["passed"] for r in items), len(items)
        bar   = "X" * p + "." * (t - p)
        print(f"    {dim:<25} {bar}  {p}/{t}")
    print()


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# columns pandas needs for its checks (GX owns the full schema check)
TEXT_COLUMNS    = ["title", "video_id", "channelId", "channelTitle"]
NUMERIC_COLUMNS = ["view_count", "likes", "comment_count"]
DATE_COLUMNS    = ["publishedAt", "trending_date"]
URL_COLUMNS     = ["thumbnail_link"]


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

df = pd.read_csv("api.csv")

# 1. Quick snapshot
quick_summary(df)

# 2. Great Expectations (declarative)
gx_results = run_gx_validation(df)

# 3. Pandas (only what GX cannot express)
validator = DataValidator()

# Completeness - blank strings (GX not-null already covers nulls)
validator.validate_no_blank_strings(df, TEXT_COLUMNS)

# Accuracy - cross-column conditional logic
validator.validate_cross_column_rules(df)

# Timeliness - runtime-relative checks GX cannot do
validator.validate_no_future_dates(df, DATE_COLUMNS)
validator.validate_date_order(df, "publishedAt", "trending_date")
validator.validate_data_freshness(df, "publishedAt", max_days_old=365)

# Outliers - statistical methods GX has no expectation for
validator.validate_outliers_iqr(df, NUMERIC_COLUMNS, multiplier=5.0)
validator.validate_outliers_zscore(df, NUMERIC_COLUMNS, threshold=5.0)

# Distribution - ratio/dominance/variance checks GX row-count/median can't cover
validator.validate_category_dominance(df, "categoryId", max_share=0.80)
validator.validate_non_zero_variance(df, NUMERIC_COLUMNS)
validator.validate_trending_ratio(df)

# Referential Integrity - parsed list-length comparison
validator.validate_count_matches_list(df, "chapter_count", "chapters")
validator.validate_count_matches_list(df, "card_count", "cards")

pandas_summary = validator.generate_report()

# 4. Unified scorecard
summarize_all(gx_results, pandas_summary)