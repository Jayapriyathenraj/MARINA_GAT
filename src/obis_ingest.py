"""
Module 1b: OBIS Species Occurrence Data Ingestion

Downloads ALL available OBIS occurrence records using pagination,
while writing each page directly to CSV instead of keeping the
entire dataset in RAM.

The final CSV remains compatible with the existing Marina-GAT pipeline.
"""

import os
import requests
import pandas as pd


def get_obis_occurrences(
    min_lat,
    max_lat,
    min_lon,
    max_lon,
    days_back=365,
    page_size=1000,
    output_csv="obis_occurrences.csv"
):
    """
    Download complete OBIS occurrence records using pagination.

    Records are written page-by-page to CSV to avoid keeping the
    complete dataset in memory.

    Returns:
        total_records
        unique_species
    """

    geometry = (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},"
        f"{min_lon} {min_lat}))"
    )

    url = "https://api.obis.org/v3/occurrence"

    start_date = (
        pd.Timestamp.now().normalize()
        - pd.Timedelta(days=days_back)
    ).strftime("%Y-%m-%d")

    # Remove previous output so a new run starts cleanly
    if os.path.exists(output_csv):
        os.remove(output_csv)

    offset = 0
    page_number = 1
    total_records = 0
    unique_species = set()

    first_write = True

    while True:

        params = {
            "geometry": geometry,
            "size": page_size,
            "offset": offset,
            "startdate": start_date,
        }

        print(
            f"Downloading OBIS page {page_number} "
            f"(offset={offset}, size={page_size})..."
        )

        try:
            response = requests.get(
                url,
                params=params,
                timeout=60
            )

            response.raise_for_status()

        except requests.RequestException as e:
            print(f"\nOBIS request failed at offset {offset}: {e}")
            print("You can safely rerun the cell to restart the download.")
            raise

        results = response.json().get("results", [])

        # No more records
        if not results:
            print("\nNo more records returned.")
            break

        page_df = pd.DataFrame(results)

        # Keep the same columns expected by the existing Marina-GAT pipeline
        keep_cols = [
            c for c in [
                "scientificName",
                "decimalLatitude",
                "decimalLongitude",
                "eventDate",
                "individualCount",
                "species"
            ]
            if c in page_df.columns
        ]

        page_df = page_df[keep_cols]

        # Remove records without usable coordinates
        if (
            "decimalLatitude" in page_df.columns
            and "decimalLongitude" in page_df.columns
        ):
            page_df = page_df.dropna(
                subset=["decimalLatitude", "decimalLongitude"]
            )

        # Track unique species without storing all occurrence records
        species_column = None

        if "species" in page_df.columns:
            species_column = "species"
        elif "scientificName" in page_df.columns:
            species_column = "scientificName"

        if species_column is not None:
            species_values = (
                page_df[species_column]
                .dropna()
                .astype(str)
                .str.strip()
            )

            unique_species.update(
                x for x in species_values
                if x
            )

        # Write this page immediately to disk
        page_df.to_csv(
            output_csv,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False

        total_records += len(page_df)

        print(
            f"  Retrieved {len(page_df)} usable records "
            f"(total so far: {total_records})"
        )

        # If fewer than page_size were returned,
        # this is the final page.
        if len(results) < page_size:
            print("\nFinal page reached.")
            break

        offset += page_size
        page_number += 1

    print("\n==========================================")
    print("          OBIS DATA SUMMARY")
    print("==========================================")
    print(f"Study period          : Last {days_back} days")
    print(f"Total usable records  : {total_records:,}")
    print(f"Unique species        : {len(unique_species):,}")
    print(f"Pages downloaded      : {page_number:,}")
    print(f"Output file           : {output_csv}")
    print("==========================================")

    return total_records, len(unique_species)


if __name__ == "__main__":

    # India East Coast
    # Tamil Nadu + Andhra Pradesh + Odisha
    # Most recent 3-months OBIS window

    total_records, unique_species = get_obis_occurrences(
        min_lat=8.0,
        max_lat=21.5,
        min_lon=78.0,
        max_lon=87.5,
        days_back=90,
        page_size=1000,
        output_csv="obis_occurrences.csv"
    )

    print("\nOBIS ingestion completed successfully.")
