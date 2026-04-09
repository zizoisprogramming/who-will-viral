import ast
import json
from datetime import datetime
from itertools import groupby

import great_expectations as gx
import numpy as np
import pandas as pd


def extract_hl_list_from_file(file_path):
    with open(file_path, encoding="utf-8") as f:
        json_data = json.load(f)

    base_codes = {
        item["snippet"]["hl"].split("-")[0].lower()
        for item in json_data.get("items", [])
    }
    return list(base_codes)

def normalize_lang(lang):
    if not lang:
        return lang
    return lang.split("-")[0].lower()

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

    context     = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(name="pandas_source")
    data_asset  = data_source.add_dataframe_asset(name="videos_asset")
    batch_def   = data_asset.add_batch_definition_whole_dataframe("batch")
    suite       = context.suites.add(gx.ExpectationSuite(name="videos_suite"))

    # ── COMPLETENESS: required columns must not be null ──────────
    for col in ["video_id", "title", "view_count", "likes", "publishedAt", "channelId", "is_trending"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )

    # ── ACCURACY: numeric bounds ──────────────────────────
    for col in ["view_count", "likes", "comment_count", "favoriteCount"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(column=col, min_value=0)
        )

    # ── ACCURACY: categoryId must be between 1 and 44 (standard) ─────────
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="categoryId", min_value=1, max_value=44)
    )

    # ── ACCURACY: duration must be valid ISO-8601 ────────────────
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToMatchRegex(
            column="duration",
            regex=r"^P(?:\d+D)?(?:T(?:\d+H)?(?:\d+M)?(?:\d+S)?)?$"
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
    # Full-row dedup: video_id repeats across dates and snapshots,
    # so only an identical row in every column is a true duplicate.
    suite.add_expectation(
        gx.expectations.ExpectCompoundColumnsToBeUnique(
            column_list=list(df.columns)
        )
    )

    # ── UNIQUENESS: key ────────────────────────────────
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(
            column='video_id'
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

    context.build_data_docs()

    try:
        context.open_data_docs()
    except Exception:
        pass

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


class DataValidator:
    def __init__(self):
        self.validation_results = []

    def _make_report(self, check_type: str, dimension: str):
        return {
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dimension":  dimension,
            "check_type": check_type,
            "passed":     True,
            "issues":     [],
            "info": []
        }

    def _save(self, report):
        self.validation_results.append(report)
        return report


    # ── CONSISTENCY (schema) ────────────────────────────────────
    def validate_schema(self, df, expected_columns, expected_types):
        """Check column presence and data types."""
        report = self._make_report("Schema", "Consistency")

        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            report["passed"] = False
            report["issues"].append(f"Missing columns: {missing_cols}")

        extra_cols = set(df.columns) - set(expected_columns)
        if extra_cols:
            report["issues"].append(f"Extra columns (not in schema): {extra_cols}")

        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    report["passed"] = False
                    report["issues"].append(
                        f"Column '{col}': expected '{expected_type}', got '{actual_type}'"
                    )

        return self._save(report)

    def validate_default_language(self, df, hl_file_path="data/youtube/hl_list.json"):
        """Default language values check against YouTube i18n language list"""
        report = self._make_report("Default Language", "Consistency")

        if "defaultLanguage" not in df.columns:
            report["issues"].append("Column 'defaultLanguage' not found")
            report["passed"] = False
            return self._save(report)

        hl_set = {
            code.split("-")[0].lower()
            for code in extract_hl_list_from_file(hl_file_path)
        } | YOUTUBE_EXTRA_LANGS

        series = df["defaultLanguage"].dropna()

        # mask for invalid rows
        invalid_mask = ~series.str.split("-").str[0].str.lower().isin(hl_set)

        invalid_rows = series[invalid_mask]
        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            report["passed"] = False
            report["issues"].append(
                f"Column 'defaultLanguage': {invalid_count} invalid row(s). "
                f"Sample values: {invalid_rows.unique().tolist()[:5]}"
            )

        return self._save(report)

    # ── COMPLETENESS ────────────────────────────────────────────
    # GX covers not-null; pandas covers blank strings which pass not-null
    def validate_no_blank_strings(self, df, text_columns):
        """Empty-string cells check"""
        report = self._make_report("Blank Strings", "Completeness")

        for col in text_columns:
            if col not in df.columns:
                continue
            n = df[col].dropna().astype(str).str.strip().eq("").sum()
            if n:
                report["passed"] = False
                report["issues"].append(f"Column '{col}': {n} blank-string values")
        return self._save(report)

    # ── ACCURACY ────────────────────────────────────────────────
    # cross-column logic

    def validate_cross_column_rules(self, df):
        """
        Business rules that span multiple columns
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
                (df["comments_disabled"].eq(True)) & (df["comment_count"] != 0)
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
        if {"contentDetails.contentRating.ytRating", "comments_disabled"}.issubset(df.columns):
            n = df[
                (df["contentDetails.contentRating.ytRating"]=="ytAgeRestricted") & (df["comments_disabled"].eq(False))
            ].shape[0]
            if n:
                report["passed"] = False
                report["issues"].append(
                    f"Age Restricted Content but comments_disabled=False: {n} rows"
                )

        return self._save(report)

    # ── TIMELINESS ──────────────────────────────────────────────
    def validate_no_future_dates(self, df, date_columns):
        """Ensuring correct dates"""
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
        """Ensuring correct date ordering"""
        report = self._make_report(
            f"Date Order ({earlier_col} <= {later_col})", "Timeliness"
        )
        if earlier_col not in df.columns or later_col not in df.columns:
            report["issues"].append(
                f"Column(s) not found: '{earlier_col}', '{later_col}'"
            )
            return self._save(report)

        t1 = pd.to_datetime(df[earlier_col], errors="coerce", utc=True).dt.date
        t2 = pd.to_datetime(df[later_col],   errors="coerce", utc=True).dt.date

        report["info"].append(f"{(t1.isna()|t2.isna()).sum()} rows skipped (null date in either column)")

        unparseable = (df[earlier_col].notna() & t1.isna()).sum() + \
                      (df[later_col].notna()   & t2.isna()).sum()
        if unparseable:
            report["issues"].append(
                f"{unparseable} non-null values could not be parsed as dates"
            )

        n  = (t1.notna() & t2.notna() & (t2 < t1)).sum()
        if n:
            report["passed"] = False
            report["issues"].append(
                f"{n} rows where '{later_col}' is earlier than '{earlier_col}'"
            )
        return self._save(report)

    # ── OUTLIERS ────────────────────────────────────────────────
    def validate_outliers_iqr(self, df, numeric_columns, multiplier=1.5):
        """outliers detection using IQR"""
        report = self._make_report(f"Outliers - IQR x{multiplier}", "Outliers")
        for col in numeric_columns:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                report["info"].append(f"Column '{col}': IQR=0, skipping (zero-inflated column)")
                continue
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
        """outliers detection using z-score"""
        report = self._make_report(f"Outliers - Z-score > {threshold}", "Outliers")
        for col in numeric_columns:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if s.std() == 0:
                continue
            n = ((df[col].dropna() - s.mean()).abs() / s.std() > threshold).sum()
            if n:
                report["passed"] = False
                report["issues"].append(
                    f"Column '{col}': {n} values with |z| > {threshold}"
                )
        return self._save(report)

    # ── DISTRIBUTION ────────────────────────────────────────────
    def validate_category_dominance(self, df, col, max_share=0.80):
        """no categry dominance"""
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
            report["info"].append(f"Top value share: {top_share:.1%}")
        return self._save(report)

    def validate_non_zero_variance(self, df, numeric_columns):
        """constant-column detection"""
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

    def validate_class_imbalance(
        self,
        df,
        col="is_trending",
        threshold=0.90,
    ):
        """class imbalance detetction"""
        report = self._make_report("Class Imbalance", "Distribution")

        if col not in df.columns:
            report["issues"].append(f"Column '{col}' not found")
            return self._save(report)

        vals = df[col].dropna().unique()
        if not set(vals).issubset({0, 1}):
            report["passed"] = False
            report["issues"].append(f"Column '{col}' contains non-binary values: {set(vals) - {0,1}}")
            return self._save(report)
        counts = df[col].value_counts(normalize=True)
        imbalance_ratio = counts.iloc[0]

        if imbalance_ratio > threshold:
            report["passed"] = False
            report["issues"].append(
                f"Severe class imbalance: dominant class = {imbalance_ratio:.1%}"
            )
        else:
            p1 = counts.get(1, 0.0)
            p0 = counts.get(0, 0.0)
            report["info"].append(f"Class balance OK: class1={p1:.1%}, class0={p0:.1%}")

        return self._save(report)


    # Relationships profile
    def validate_correlation(self, df, numeric_columns, corr_threshold=0.7, method='pearson'):
        """validating numeric columns correlation"""
        report = self._make_report(f"Correlation ({method})", "Relationships")
        found_issue = False

        for i in range(len(numeric_columns)):
            if numeric_columns[i] not in df.columns.tolist():
                continue
            for j in range(i + 1, len(numeric_columns)):
                if numeric_columns[j] not in df.columns.tolist():
                    continue
                corr = df[numeric_columns[i]].corr(df[numeric_columns[j]],method=method)
                if abs(corr) > corr_threshold:
                    report["issues"].append(
                        f"Correlation {corr:.2f} > {corr_threshold} "
                        f"between '{numeric_columns[i]}' and '{numeric_columns[j]}'"
                    )
                    report["passed"] = False
                    found_issue = True

        if not found_issue:
            report["info"].append(f"All correlations are below {corr_threshold}")

        return self._save(report)

    def validate_skew(self, df, numeric_columns, skew_threshold=1):
        """validating numeric columns skewness"""
        report = self._make_report("Skewness", "Relationships")
        found_issue = False

        for col in numeric_columns:
            if col not in df.columns.tolist():
                continue

            skew_val = df[col].skew()

            if abs(skew_val) > skew_threshold:
                report["issues"].append(
                    f"Skewness {skew_val:.2f} exceeds threshold {skew_threshold} in '{col}'"
                )
                report["passed"] = False
                found_issue = True

        if not found_issue:
            report["info"].append(f"All skewness values are within ±{skew_threshold}")

        return self._save(report)


    # ── REFERENTIAL INTEGRITY ────────────────────────────────────
    def validate_count_matches_list(self, df, count_col, list_col):
        """matching list columns with count columns"""
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
        parse_errors  = (actual == -1).sum()
        mismatch = (
            actual.ge(0) &
            (df[count_col].fillna(0).astype(int) != actual)
        ).sum()

        if parse_errors:
            report["passed"] = False
            report["issues"].append(
                f"{parse_errors} rows in '{list_col}' could not be parsed as a list"
            )

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


TEXT_COLUMNS    = ["title", "video_id", "channelId", "channelTitle"]
NUMERIC_COLUMNS = ["view_count", "likes", "comment_count", "categoryId", "favoriteCount", "card_count", "chapter_count"]
CATEGORICAL_COLUMNS = ['projection']
DATE_COLUMNS    = ["publishedAt", "trending_date"]
URL_COLUMNS     = ["thumbnail_link"]

EXPECTED_COLUMNS = [
    'video_id', 'title', 'publishedAt', 'channelId', 'channelTitle',
    'categoryId', 'trending_date', 'tags', 'view_count', 'likes',
    'comment_count', 'thumbnail_link', 'description', 'is_trending',
    'defaultLanguage', 'duration', 'dimension', 'definition', 'caption',
    'licensedContent', 'projection', 'embeddable', 'madeForKids',
    'favoriteCount', 'contentDetails.regionRestriction.blocked',
    'contentDetails.regionRestriction.allowed',
    'contentDetails.contentRating.ytRating', 'chapter_count',
    'chapters', 'playability_status',
    'supports_miniplayer', 'card_count', 'cards', 'is_verified',
    'badge_labels', 'comments_disabled', 'has_paid_promotion'
]

EXPECTED_TYPES = {
    "video_id":      "object",
    "title":         "object",
    "view_count":    "int64",
    "likes":         "int64",
    "comment_count": "int64",
    "is_trending":   "int64",
    "publishedAt":   "object",
}

YOUTUBE_EXTRA_LANGS = {
    "yue", "yue-hk", "bh", "bho", "mai", "sat", "bgc",
    "chr", "mni", "vro", "ase", "mo", "bi", "und", "zxx", "sdp"
}

df = pd.read_csv("data/youtube/dataset.csv")

# 1. Quick snapshot
quick_summary(df)

# 2. Great Expectations (declarative)
gx_results = run_gx_validation(df)

# 3. Pandas (only what GX cannot express)
validator = DataValidator()

# Consistency
validator.validate_schema(df, EXPECTED_COLUMNS, EXPECTED_TYPES)
validator.validate_default_language(df)

# Completeness
validator.validate_no_blank_strings(df, TEXT_COLUMNS)

# Accuracy - cross-column conditional logic
validator.validate_cross_column_rules(df)

# Timeliness
validator.validate_no_future_dates(df, DATE_COLUMNS)
validator.validate_date_order(df, "publishedAt", "trending_date")

# Outliers
validator.validate_outliers_iqr(df, NUMERIC_COLUMNS, multiplier=1.5)
validator.validate_outliers_zscore(df, NUMERIC_COLUMNS, threshold=3.0)

# Distribution
validator.validate_category_dominance(df, "categoryId", max_share=0.80)
validator.validate_non_zero_variance(df, NUMERIC_COLUMNS)
validator.validate_class_imbalance(df, "is_trending", threshold=0.90)

# Relationships
validator.validate_correlation(df, NUMERIC_COLUMNS, 0.6)
validator.validate_correlation(df, NUMERIC_COLUMNS, 0.6, method="spearman")
validator.validate_skew(df,NUMERIC_COLUMNS, 0.6)

# Referential Integrity
validator.validate_count_matches_list(df, "chapter_count", "chapters")
validator.validate_count_matches_list(df, "card_count", "cards")

pandas_summary = validator.generate_report()

# 4. Unified scorecard
summarize_all(gx_results, pandas_summary)
