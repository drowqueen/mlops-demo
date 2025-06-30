# scripts/clean_data.py
import pandas as pd
import numpy as np


class AmesDataCleaner:
    def __init__(self, df):
        self.df = df.copy()

    def add_total_sf(self, df):
        """
        Total square footage = 1st Flr SF + 2nd Flr SF + Total Bsmt SF
        """
        df["totalSF"] = df["1st Flr SF"] + df["2nd Flr SF"] + df["Total Bsmt SF"]
        return df

    def add_finished_sf(self, df):
        """
        Finished square footage = totalSF - unfinished basement space (Bsmt Unf SF)
        """
        df["finishedSF"] = (
            df["1st Flr SF"]
            + df["2nd Flr SF"]
            + df["Total Bsmt SF"]
            - df["Bsmt Unf SF"]
        )
        return df

    def add_high_quality_sf(self, df):
        """
        High quality square footage = totalSF - low quality finished space (Low Qual Fin SF)
        """
        df["qualitySF"] = (
            df["1st Flr SF"]
            + df["2nd Flr SF"]
            + df["Total Bsmt SF"]
            - df["Low Qual Fin SF"]
        )
        return df

    def add_total_bath(self, df):
        """
        Total bathrooms = FullBath + HalfBath
        """
        df["TotalBath"] = df["Full Bath"] + 0.5 * df["Half Bath"]
        return df

    def add_age(self, df):
        """
        age = Year Sold - Year Built
        """
        df["age"] = df["Yr Sold"] - df["Year Built"]
        return df

    def add_remodeled_age(self, df):
        df["remodeled_age"] = df["Yr Sold"] - df["Year Remod/Add"]
        return df

    def add_has_pool(self, df):
        if "Pool Area" in df.columns:
            df["has_pool"] = (df["Pool Area"] > 0).astype(int)
        else:
            df["has_pool"] = 0
        return df

    def add_has_fireplace(self, df):
        if "Fireplaces" in df.columns:
            df["has_fireplace"] = (df["Fireplaces"] > 0).astype(int)
        else:
            df["has_fireplace"] = 0
        return df

    def add_has_garage(self, df):
        if "Garage Cars" in df.columns:
            df["has_garage"] = (df["Garage Cars"] > 0).astype(int)
        else:
            df["has_garage"] = 0
        return df

    def add_is_new(self, df):
        df["is_new"] = (df["Year Built"] >= df["Yr Sold"] - 5).astype(int)
        return df

    def add_lot_ratio(self, df):
        df["lot_ratio"] = df["Lot Area"] / (df["Gr Liv Area"] + 1)
        return df

    def add_porch_area(self, df):
        porch_cols = ["Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch"]
        df["porch_area"] = sum([df.get(col, 0) for col in porch_cols])
        return df

    def add_age_buckets(self, df):
        """
        Create categorical age buckets for the property age.
        Buckets example (years):
            0-10: 'new'
            11-20: 'recent'
            21-40: 'mid_age'
            41-60: 'old'
            60+: 'very_old'
        """
        bins = [0, 10, 20, 40, 60, 200]
        labels = ["new", "recent", "mid_age", "old", "very_old"]
        df["age_bucket"] = pd.cut(
            df["age"], bins=bins, labels=labels, right=True, include_lowest=True
        )
        return df

    def add_bed_bath_ratio(self, df):
        """
        Ratio of bedrooms to total bathrooms (avoid division by zero)
        """
        df["bed_bath_ratio"] = df["Bedroom AbvGr"] / df["TotalBath"].replace(0, 0.1)
        return df

    def add_total_rooms(self, df):
        """
        Total rooms above ground approximation:
        Bedrooms + bathrooms + kitchen above ground
        """
        # We don't have kitchen count, use Bedroom AbvGr + TotalBath + 1 (for kitchen assumed)
        df["total_rooms"] = df["Bedroom AbvGr"] + df["TotalBath"] + 1
        return df

    def add_recently_remodeled(self, df):
        """
        Binary feature if remodeled in last 5 years before sale
        """
        df["recently_remodeled"] = (
            (df["Year Remod/Add"] >= (df["Yr Sold"] - 5)) & (df["Year Remod/Add"] != 0)
        ).astype(int)
        return df

    def add_log_transforms(self, df):
        """
        Add log(1 + x) transformed features for skewed numerical columns.
        """
        skewed_cols = [
            "Gr Liv Area",
            "Lot Area",
            "totalSF",
            "finishedSF",
            "qualitySF",
            "porch_area",
        ]
        for col in skewed_cols:
            if col in df.columns:
                df[f"log_{col}"] = (df[col] + 1).apply(np.log)
        return df

    def run(self):
        """Run all feature engineering steps and combine results."""
        # Run all feature engineering steps and combine results
        df = self.df.copy()
        df = self.add_total_sf(df)
        df = self.add_finished_sf(df)
        df = self.add_total_bath(df)
        df = self.add_high_quality_sf(df)
        df = self.add_age(df)
        df = self.add_remodeled_age(df)
        df = self.add_has_pool(df)
        df = self.add_has_fireplace(df)
        df = self.add_has_garage(df)
        df = self.add_is_new(df)
        df = self.add_lot_ratio(df)
        df = self.add_porch_area(df)
        df = self.add_age_buckets(df)
        df = self.add_bed_bath_ratio(df)
        df = self.add_total_rooms(df)
        df = self.add_recently_remodeled(df)
        df = self.add_log_transforms(df)
        return df


def clean_ames_data(input_path, output_path):
    """Clean and engineer features for the Ames Housing dataset."""
    df = pd.read_csv(input_path)
    print(f"Original data shape: {df.shape}")

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Fill missing values using exact column names from your list
    df["Garage Cars"] = df["Garage Cars"].fillna(0)
    df["Garage Area"] = df["Garage Area"].fillna(0)
    df["Total Bsmt SF"] = df["Total Bsmt SF"].fillna(0)
    df["1st Flr SF"] = df["1st Flr SF"].fillna(0)
    df["2nd Flr SF"] = df["2nd Flr SF"].fillna(0)
    df["Bsmt Unf SF"] = df["Bsmt Unf SF"].fillna(0)
    df["Low Qual Fin SF"] = df["Low Qual Fin SF"].fillna(0)
    df["Full Bath"] = df["Full Bath"].fillna(0)
    df["Half Bath"] = df["Half Bath"].fillna(0)
    df["Pool Area"] = df["Pool Area"].fillna(0)
    df["Fireplaces"] = df["Fireplaces"].fillna(0)
    df["Lot Area"] = df["Lot Area"].fillna(0)
    df["Open Porch SF"] = df["Open Porch SF"].fillna(0)
    df["Enclosed Porch"] = df["Enclosed Porch"].fillna(0)
    df["3Ssn Porch"] = df["3Ssn Porch"].fillna(0)
    df["Screen Porch"] = df["Screen Porch"].fillna(0)

    # Select base features with exact names as in your dataset
    base_features = [
        "Overall Qual",
        "Year Built",
        "Year Remod/Add",
        "Garage Cars",
        "Garage Area",
        "Bedroom AbvGr",
        "Yr Sold",
        "SalePrice",
        "Full Bath",
        "Half Bath",
        "1st Flr SF",
        "2nd Flr SF",
        "Total Bsmt SF",
        "Bsmt Unf SF",
        "Low Qual Fin SF",
        "Pool Area",
        "Fireplaces",
        "Lot Area",
        "Open Porch SF",
        "Enclosed Porch",
        "3Ssn Porch",
        "Screen Porch",
        "Gr Liv Area",
    ]

    df = df[base_features]

    # Remove houses with extremely low or high SalePrice
    df = df[(df["SalePrice"] >= 5000) & (df["SalePrice"] <= 500000)]
    # Remove houses with abnormally large living area (greater than 4000 sqft)
    df = df[df["Gr Liv Area"] <= 4000]

    # Run feature engineering
    cleaner = AmesDataCleaner(df)
    df = cleaner.run()

    # Drop rows with missing critical data or target
    df = df.dropna(
        subset=[
            "Overall Qual",
            "Year Built",
            "Year Remod/Add",
            "Garage Cars",
            "Garage Area",
            "Bedroom AbvGr",
            "Yr Sold",
            "SalePrice",
        ]
    )

    # Convert any boolean dummies to integers (True → 1, False → 0)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"Data shape after cleaning and encoding: {df.shape}")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean Ames Housing data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/AmesHousing.csv",
        help="Path to raw input CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ames_cleaned.csv",
        help="Path to output cleaned CSV",
    )
    args = parser.parse_args()

    clean_ames_data(input_path=args.input, output_path=args.output)
