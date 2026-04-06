import pandas as pd
import numpy as np
import great_expectations as gx

class CustomValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = []

    def _add_result(self, name, passed, failed_rows):
        self.results.append({
            "rule": name,
            "passed": passed,
            "failed_count": len(failed_rows),
            "sample_failed": failed_rows.head(3).to_dict(orient="records")
        })

    # Rule 1:
    def validate_trending_date(self):
        invalid = self.df[
            (self.df["is_trending"] == 1) &
            (self.df["trending_date"].isna())
        ]
        self._add_result(
            "If is_trending=1 → trending_date must not be null",
            invalid.empty,
            invalid
        )

    # Rule 2:
    def validate_comments_disabled(self):
        invalid = self.df[
            (self.df["comments_disabled_y"] == True) &
            (self.df["comment_count_x"] != 0)
        ]
        self._add_result(
            "If comments_disabled=True → comment_count must be 0",
            invalid.empty,
            invalid
        )

    def run_all(self):
        self.validate_trending_date()
        self.validate_comments_disabled()
        return self.results

def run_validation(df: pd.DataFrame):
    # Step 1: Create an in-memory GX context (no files written to disk)
    context = gx.get_context(mode="ephemeral")

    # Step 2: Connect GX to your pandas DataFrame
    data_source = context.data_sources.add_pandas(name="my_pandas_source")
    data_asset = data_source.add_dataframe_asset(name="videoss_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("my_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    # Step 3: Create an Expectation Suite (a named collection of rules)
    suite = context.suites.add(gx.ExpectationSuite(name="videos_validation_suite"))

    # ── DIMENSION 1: ACCURACY (business rules) ───────────────────────────
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="likes_x", min_value=0
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="view_count", min_value=0
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="comment_count", min_value=0
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToMatchRegex(column="duration", regex=r"^P(?:(\d+)D)?T(?=\d+[HMS])(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$")
    )

    # ── DIMENSION 2: COMPLETENESS ─────────────────────────────────────────
    important_cols = [
        "video_id", "title", "view_count", "likes",
        "publishedAt", "channelId"
    ]
    for col in important_cols:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    # ── DIMENSION 3: UNIQUENESS ───────────────────────────────────────────
    for col in ["video_id"]:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column=col))

    suite.add_expectation(
        gx.expectations.ExpectCompoundColumnsToBeUnique(
            column_list=["video_id", "trending_date"]
        )
    )

    # ── DIMENSION 4: CONSISTENCY (schema) ────────────────────────────────
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=['video_id', 'title', 'publishedAt', 'channelId',
       'channelTitle', 'categoryId', 'trending_date', 'tags',
       'view_count', 'likes', 'comment_count', 'thumbnail_link', 'description', 'is_trending', 'defaultLanguage',
       'duration', 'dimension', 'definition', 'caption', 'licensedContent',
       'projection', 'embeddable', 'madeForKids',
       'favoriteCount',
       'contentDetails.regionRestriction.blocked',
       'contentDetails.regionRestriction.allowed',
       'contentDetails.contentRating.ytRating', 'url', 'chapter_count',
       'chapters', 'playability_status', 'is_age_restricted',
       'supports_miniplayer', 'card_count', 'cards', 'is_verified',
       'badge_labels', 'comments_disabled', 'has_paid_promotion']
        )
    )

    # ── DIMENSION 5: CATEGORICAL (allowed values) ─────────────────────────
    # default language , dimension, definition, projection
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="defaultLanguage", value_set=[
            "en", "en-US", "en-GB",
            "ar", "ar-EG", "ar-SA",
            "es", "es-ES", "es-419",
            "fr", "fr-FR",
            "de",
            "it",
            "pt", "pt-BR",
            "ru",
            "hi",
            "id",
            "ja",
            "ko",
            "zh", "zh-CN", "zh-TW",
            "tr",
            "nl",
            "pl",
            "sv",
            "th",
            "vi"
        ]
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="dimension", value_set=["2d"]
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="definition", value_set=["hd","sd"]
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="projection", value_set=["rectangular"]
        )
    )

    # ── DIMENSION 6: DISTRIBUTION ─────────────────────────────────────────
    # suite.add_expectation(
    #     gx.expectations.ExpectColumnMeanToBeBetween(
    #         column="age", min_value=30, max_value=60
    #     )
    # )
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=100, max_value=100000)
    )

    # Step 4: Create a Validation Definition (links batch to suite)
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(name="videos_validation", data=batch_def, suite=suite)
    )

    # Step 5: Run validation
    results = validation_def.run(batch_parameters={"dataframe": df})

    custom_validator = CustomValidator(df)
    custom_results = custom_validator.run_all()

    # Step 6: Print readable report
    _print_report(results)
    _print_custom_report(custom_results)
    summarize_all(results, custom_results)

    context.build_data_docs()
    context.open_data_docs()

    return results

def _print_custom_report(custom_results):
    print("\n" + "=" * 58)
    print("    CUSTOM BUSINESS RULES VALIDATION")
    print("=" * 58)

    for res in custom_results:
        status = "PASS" if res["passed"] else "FAIL"

        print(f"\n[{status}] {res['rule']}")
        print(f"   Failed Rows : {res['failed_count']}")

        if not res["passed"]:
            print(f"   Sample : {res['sample_failed']}")

def _print_report(results):
    """Print a clean summary of GX v1.x validation results."""

    # GX 1.x stores results differently from 0.x
    result_dict = results.describe()
    success = results.success

    print("=" * 58)
    print("    DATA VALIDATION REPORT  (Great Expectations v1.x)")
    print("=" * 58)
    print(f"  Overall Result : {'PASSED' if success else 'FAILED'}")
    print("=" * 58)

    # Iterate through each expectation result
    for exp_result in results.results:
        exp_type = exp_result.expectation_config.type
        col = exp_result.expectation_config.kwargs.get("column", "table-level")
        passed = exp_result.success
        status = "PASS" if passed else "FAIL"

        print(f"\n[{status}] {exp_type}")
        print(f"   Column : {col}")

        if not passed and exp_result.result:
            r = exp_result.result
            if r.get("unexpected_count"):
                print(f"   Issues : {r['unexpected_count']} unexpected values")
            if r.get("partial_unexpected_list"):
                print(f"   Sample : {r['partial_unexpected_list'][:3]}")

    print("\n" + "=" * 58)

def summarize_all(gx_results, custom_results):
    total_gx = len(gx_results.results)
    passed_gx = sum(r.success for r in gx_results.results)

    total_custom = len(custom_results)
    passed_custom = sum(r["passed"] for r in custom_results)

    print("\n" + "=" * 58)
    print(" FINAL VALIDATION SUMMARY ")
    print("=" * 58)
    print(f"GX: {passed_gx}/{total_gx} passed")
    print(f"Custom: {passed_custom}/{total_custom} passed")
    print(f"Overall: {(passed_gx + passed_custom)}/{(total_gx + total_custom)} passed")

# ─────────────────────────────────────────────
# Load DATA
# ─────────────────────────────────────────────

# df = pd.read_csv('../example.csv', parse_dates=['signup_date'])
df = pd.read_csv('example.csv')

# ─────────────────────────────────────────────
# RUN VALIDATION
# ─────────────────────────────────────────────
run_validation(df)
