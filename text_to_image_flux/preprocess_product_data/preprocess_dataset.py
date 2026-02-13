"""
Fashion Product Dataset Preprocessing Pipeline
===============================================
This script performs complete preprocessing on the fashion product dataset:
1. Extracts brand names from display names
2. Filters out kids/teens/boys/girls products
3. Normalizes brand names (removes color suffixes, consolidates variations)

Input: data.csv
Output: data_preprocessed.csv
"""

import csv
import os
import re
from collections import defaultdict

from src.constants import DATASET_DIR

# Configuration
INPUT_FILE = os.path.join(DATASET_DIR, "fashion-product", "data.csv")
OUTPUT_FILE = os.path.join(DATASET_DIR, "fashion-product", "data_preprocessed.csv")

# Keywords to filter out
EXCLUDE_KEYWORDS = [
    "Kids",
    "Teens",
    "Boy's",
    "Girl's",
    "Boys",
    "Girls",
    "Infant",
    "Infants",
    "Unisex",
]

# Description keywords to filter out
EXCLUDE_DESCRIPTION_KEYWORDS = [
    "Model statistics",
    "Model Statistics",
    "Model's Statistics",
    "Model's Statistics",
]

# Brand normalization rules
BRAND_NORMALIZATIONS = {
    # Remove color suffixes
    r"^(.*?)\s+(Black|White|Navy Blue|Pink|Red|Blue|Green|Grey|Gray|Brown|Purple|Yellow|Orange|Maroon|Tan)$": r"\1",
    # ADIDAS variations
    r"^ADIDAS.*": "ADIDAS",
    r"^Adidas.*": "Adidas",
    # CASIO variations
    r"^CASIO\s+(BABY-G|Baby-G|EDIFICE|Edifice|ENTICER|Enticer|G-SHOCK|G-Shock|OUTDOOR|Outdoor|SHEEN|Sheen|YOUTH|Youth.*)$": "CASIO",
    r"^Casio\s+(BABY-G|Baby-G|EDIFICE|Edifice|ENTICER|Enticer|Enticerr|G-SHOCK|G-Shock|Outdoor|SHEEN|Sheen|YOUTH|Youth.*)$": "Casio",
    # Nike variations
    r"^Nike\s+(Fragrances|JDI Pink|Light Green)$": "Nike",
    # Puma variations
    r"^Puma\s+(Deccan Chargers|Rajasthan Royals)$": "Puma",
    # United Colors of Benetton
    r"^United Colors [Oo]f Benetton$": "United Colors of Benetton",
    r"^United colors of benetton$": "United Colors of Benetton",
    # Hugo Boss variations
    r"^Hugo Boss\s+(XX|XY)$": "Hugo Boss",
    r"^Hugo\s+(XY)$": "Hugo Boss",
    r"^Hugo$": "Hugo Boss",
    # BIBA variations
    r"^BIBA OUTLET$": "BIBA",
    r"^Biba Outlet$": "Biba",
    # Other specific normalizations
    r"^4711\s+(Atomizer|Kolnisch Wasser|Vapo)$": "4711",
    r"^AND by Anita Dongre$": "AND",
    r"^Allen Solly\s+(Woman)$": "Allen Solly",
    r"^American Tourister$": "American Tourister",
    r"^Arrow\s+(New York|Sport|Sports|Woman)$": "Arrow",
    r"^Artengo\s+522 C$": "Artengo",
    r"^Avirate\s+Black & Purple$": "Avirate",
    r"^Baldessarini\s+(Ambre|Delmar|HUGO BOSS)$": "Baldessarini",
    r"^Boss In Motion$": "Boss",
    r"^Buckaroo Jefe$": "Buckaroo",
    r"^Bulchee Ufficio$": "Bulchee",
    r"^Burberry London$": "Burberry",
    r"^Calvin Klein Innerwear$": "Calvin Klein",
    r"^Carlton London$": "Carlton",
    r"^Classic Polo$": "Classic Polo",
    r"^Cobblerz Blue$": "Cobblerz",
    r"^Colour [Mm]e$": "Colour Me",
    r"^Coolers by Liberty$": "Coolers",
    r"^Crocs Patricia$": "Crocs",
    r"^DC Comics$": "DC",
    r"^DKNY Pure$": "DKNY",
    r"^David Beckham\s+(Instinct|Intense Instinct|Intimately|Signature)$": "David Beckham",
    r"^Decathlon Profilter$": "Decathlon",
    r"^Denim Original$": "Denim",
    r"^Do [Uu] speak (Green|Cream|green)$": "Do U Speak Green",
    r"^Dunhill\s+(Desire|Fresh|Pure|Pursuit)$": "Dunhill",
    r"^Ed Hardy\s+.*$": "Ed Hardy",
    r"^FCUK Underwear$": "FCUK",
    r"^Fastrack Titan$": "Fastrack",
    r"^Flying Machine$": "Flying Machine",
    r"^Forever New Women Creme de$": "Forever New",
    r"^Formula 1\s+(Go|Gold|Start)$": "Formula 1",
    r"^French Connection$": "French Connection",
    r"^GUESS\s+(Seductive|by Marciano)$": "GUESS",
    r"^Guess\s+(Marciano|Natural)$": "Guess",
    r"^Gini and Jony$": "Gini & Jony",
    r"^Giorgio Beverly Hills$": "Giorgio Armani",
    r"^Gliders by Liberty$": "Gliders",
    r"^Hidekraft Casual$": "Hidekraft",
    r"^Ice Watch$": "Ice Watch",
    r"^Indigo Nation$": "Indigo Nation",
    r"^Ivory Tag$": "Ivory",
    r"^J\. Del Pozo$": "J.Del Pozo",
    r"^J\.Del Pozo\s+(Ambar|Halloween|Halloween Kiss Sexy|Quasar)$": "J.Del Pozo",
    r"^Jaguar\s+(Casual|Classic|Prestige)$": "Jaguar",
    r"^Jealous 21$": "Jealous 21",
    r"^Jockey\s+(24 x 7|COMFORT PLUS|COMFPLUS|CSM|ELANCE|GOLDEDN|LCESCBRA|MC|MODERN CLASSIC|Modern Classic|Pack of 2|SP-IW|SP-OW|SPORT|SPORT PERFORMANCE|Sport Performance|ZONE|ZONE STRETCH)$": "Jockey",
    r"^John (Lenon|Miller|Players)$": r"John \1",
    r"^Just Cavalli$": "Just Cavalli",
    r"^Just Natural$": "Just Natural",
    r"^Kalenji Ekiden 50$": "Kalenji",
    r"^Kiara Black$": "Kiara",
    r"^Latin Quarters$": "Latin Quarters",
    r"^Lee Cooper$": "Lee Cooper",
    r"^Lino Perros.*$": "Lino Perros",
    r"^Little Miss Intimates$": "Little Miss Intimates",
    r"^Lotus Herbals.*$": "Lotus Herbals",
    r"^Love Passport.*$": "Love Passport",
    r"^M [Tt]v$": "MTV",
    r"^Mark Taylor$": "Mark Taylor",
    r"^Marvel Comics$": "Marvel",
    r"^Maxima\s+(Aqua|Attivo|Ssteele|Steel)$": "Maxima",
    r"^Miss Sixty$": "Miss Sixty",
    r"^Mumbai (Indians|Slang)$": r"Mumbai \1",
    r"^Nautica\s+(Blue|Pure|Voyage|White Sail)$": "Nautica",
    r"^New Balance\s+(M364|MC900|MR310|MW631)$": "New Balance",
    r"^New Hide$": "New Hide",
    r"^Newhide.*$": "Newhide",
    r"^Numero Uno.*$": "Numero Uno",
    r"^Paris Hilton\s+.*$": "Paris Hilton",
    r"^Park Avenue$": "Park Avenue",
    r"^Perry Ellis.*$": "Perry Ellis",
    r"^Peter England\s+Elements$": "Peter England",
    r"^Pink Floyd$": "Pink Floyd",
    r"^Playboy.*$": "Playboy",
    r"^Police\s+(Cosmopolitian|Dark|Passion|Pure|Titanium Wings|Wings)$": "Police",
    r"^Q&Q\s+(Attractive|Dynamiq|Superior)$": "Q&Q",
    r"^Red Tape.*$": "Red Tape",
    r"^Red Chief$": "Red Chief",
    r"^Red Rose$": "Red Rose",
    r"^Reebok\s+(Knit Rib|Reebounce|Reecharge|Reefresh|Reegame|Reelive|Reeload|Reenergize|Reeplay|Reesport|Track Pants)$": "Reebok",
    r"^Regent Polo Club$": "Regent Polo Club",
    r"^Rocky S$": "Rocky S",
    r"^Royal Diadem$": "Royal Diadem",
    r"^SDL by Sweet Dreams$": "SDL",
    r"^Satya Paul$": "Satya Paul",
    r"^Scullers [Ff]or Her$": "Scullers",
    r"^Solognac Inv 50$": "Solognac",
    r"^Spice Art$": "Spice Art",
    r"^Tabac Original.*$": "Tabac",
    r"^Taylor of London$": "Taylor of London",
    r"^Timex\s+(Empera|Expedition|Helix|Intelligent Quartz)$": "Timex",
    r"^Titan\s+(Edge|Obahu|Raga)$": "Titan",
    r"^Tonino Lamborghini.*$": "Tonino Lamborghini",
    r"^Tous\s+.*$": "Tous",
    r"^Tribord Profilter Red$": "Tribord",
    r"^Turtle\s+(Check|Solid|Stripes)$": "Turtle",
    r"^U\.S\. Polo Assn\.\s+Denim Co\.$": "U.S. Polo Assn.",
    r"^U\.S\.Polo Assn\.?$": "U.S. Polo Assn.",
    r"^Van Heusen$": "Van Heusen",
    r"^Vero Moda.*$": "Vero Moda",
    r"^Wild [Ss]tone$": "Wild Stone",
    r"^Wills\s+(Classic|Lifestyle|Lifetsyle|Sport)$": "Wills Lifestyle",
    r"^Wrangler\s+(Canvas|Leather|Plate|Single|Stitch|Stud|Textured)$": "Wrangler",
    r"^Black [Cc]offee$": "Black Coffee",
}


def extract_brand(display_name):
    """Extract brand name from the display name."""
    if not display_name:
        return ""

    # Common patterns: "Brand Men/Women/Boys/Girls/Kids/Unisex Product"
    keywords = ["Men", "Women", "Boys", "Girls", "Kids", "Unisex", "Men's", "Women's"]

    for keyword in keywords:
        if keyword in display_name:
            parts = display_name.split(keyword)
            brand = parts[0].strip()
            if brand:
                return brand

    # If no keyword found, return first word as brand
    words = display_name.split()
    if len(words) > 0:
        return words[0]

    return ""


def should_exclude(display_name, brand, description):
    """Check if the row should be excluded based on keywords."""
    # Check display name and brand for kids/teens keywords
    text = display_name + " " + brand
    for keyword in EXCLUDE_KEYWORDS:
        if keyword in text:
            return True

    # Check description for model statistics
    for keyword in EXCLUDE_DESCRIPTION_KEYWORDS:
        if keyword in description:
            return True

    return False


def normalize_brand(brand):
    """Apply normalization rules to brand names."""
    if not brand or not brand.strip():
        return brand

    brand = brand.strip()

    # Apply each normalization rule
    for pattern, replacement in BRAND_NORMALIZATIONS.items():
        brand = re.sub(pattern, replacement, brand)

    return brand


def main():
    """Main preprocessing pipeline."""
    print("=" * 80)
    print("Fashion Product Dataset Preprocessing Pipeline")
    print("=" * 80)

    # Statistics
    total_rows = 0
    excluded_rows = 0
    excluded_kids = 0
    excluded_model_stats = 0
    brands_extracted = 0
    brands_normalized = 0

    # Process data
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(
        OUTPUT_FILE, "w", encoding="utf-8", newline=""
    ) as outfile:

        reader = csv.DictReader(infile)
        fieldnames = ["image", "description", "display name", "brand", "category"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        for row in reader:
            total_rows += 1

            # Step 1: Extract brand name
            brand = extract_brand(row["display name"])
            brands_extracted += 1

            # Step 2: Filter out kids/teens products
            if should_exclude(row["display name"], brand, row["description"]):
                excluded_rows += 1
                # Track specific exclusion reasons
                text = row["display name"] + " " + brand
                is_kids = any(keyword in text for keyword in EXCLUDE_KEYWORDS)
                has_model_stats = any(
                    keyword in row["description"]
                    for keyword in EXCLUDE_DESCRIPTION_KEYWORDS
                )

                if is_kids:
                    excluded_kids += 1
                if has_model_stats:
                    excluded_model_stats += 1
                continue

            # Step 3: Normalize brand name
            original_brand = brand
            brand = normalize_brand(brand)
            if original_brand != brand:
                brands_normalized += 1

            # Write processed row
            writer.writerow(
                {
                    "image": row["image"],
                    "description": row["description"],
                    "display name": row["display name"],
                    "brand": brand,
                    "category": row["category"],
                }
            )

    # Calculate final statistics
    final_rows = total_rows - excluded_rows

    # Count unique brands
    unique_brands = set()
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["brand"].strip():
                unique_brands.add(row["brand"].strip())

    # Print summary
    print("\nProcessing Summary:")
    print("-" * 80)
    print(f"Total rows processed:           {total_rows:>10,}")
    print(f"Rows excluded (total):          {excluded_rows:>10,}")
    print(f"  - Kids/Teens/Boys/Girls:      {excluded_kids:>10,}")
    print(f"  - With Model statistics:      {excluded_model_stats:>10,}")
    print(f"Final rows (filtered):          {final_rows:>10,}")
    print(f"Brand names normalized:         {brands_normalized:>10,}")
    print(f"Unique brands in output:        {len(unique_brands):>10,}")
    print("-" * 80)
    print(f"\nOutput file created: {OUTPUT_FILE}")
    print("=" * 80)

    # Show top 20 brands
    brand_counts = defaultdict(int)
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            brand = row["brand"].strip()
            if brand:
                brand_counts[brand] += 1

    print("\nTop 20 Brands by Product Count:")
    print("-" * 80)
    for i, (brand, count) in enumerate(
        sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:20], 1
    ):
        print(f"{i:2}. {brand:40} {count:>6,} products")
    print("=" * 80)


if __name__ == "__main__":
    main()
