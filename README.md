# Corporate Values Analysis Project

## Overview
This project analyzes corporate values and culture across different industries by examining keyword matches related to core values (Innovation, Integrity, Quality, Respect, and Teamwork) using Thai language word embeddings and natural language processing techniques.

## Features
- Comprehensive keyword match analysis across five core values
- Thai language text processing and word tokenization
- Word embedding analysis using Thai2fit word vectors
- Industry-specific performance metrics and comparisons
- Advanced data visualization including:
  - Distribution of keyword matches by industry
  - Industry score heatmaps
  - Performance overview scatter plots
  - Score distribution box plots
  - Total matches analysis

## Dependencies
```python
pandas
matplotlib
seaborn
numpy
xlsxwriter # For Excel report generation
pythainlp # For Thai language processing
gensim # For word embeddings
sklearn # For dimensionality reduction
tqdm # For progress bars
pickle # For data serialization
```

## Data Preparation
1. Word Tokenization and Counting:
```python
from pythainlp.corpus.common import thai_stopwords
import pickle

# Load and process data
data = pickle.load(open('entriesOct.pickle','rb'))
df = data[['Tokenized']].explode('Tokenized')

# Count word frequencies
all_word = df.groupby('Tokenized').size().reset_index()
all_word.columns = ['word','cnt']

# Mark stop words
all_word['is_stop_word'] = all_word['word'].apply(lambda x: x in thai_stopwords())

# Save processed data
all_word.sort_values('cnt', ascending=False).to_pickle('word_count.pickle')
```

2. Word Embedding Model:
```python
# Load Thai word vectors
model = word_vector.WordVector(model_name="thai2fit_wv").get_model()

# Create dictionary of word vectors
thai2dict = {}
for word in model.index2word:
    thai2dict[word] = model[word]
thai2vec = pd.DataFrame.from_dict(thai2dict, orient='index')
```

## Project Structure
- Input files:
  - `entriesOct.pickle`: Original tokenized text data
  - `word_count.pickle`: Processed word frequency counts
  - `thai_seed_words.json`: Seed words for cultural values
- Generated output files:
  - `word2vec_result.pickle`: Word embedding analysis results
  - `summary_result.csv`: Final analysis results
  - `keyword_analysis.png`: Overview analysis visualizations
  - `industry_analysis_enhanced_large.png`: Detailed industry analysis visualizations
  - `industry_analysis_detailed.xlsx`: Detailed Excel report
  - Individual plot files:
    - `keyword_distribution.png`
    - `industry_scores_heatmap.png`
    - `total_matches.png`
    - `score_distribution.png`
    - `performance_overview.png`

## Key Functions

### Word Vector Analysis
- Word similarity calculation using Thai2fit word vectors
- Cosine similarity measurements for related words
- Custom distance metrics for cultural value analysis

### `create_industry_plots(df)`
Creates and saves individual visualization plots for different aspects of the industry analysis:
- Keyword distribution
- Industry score heatmap
- Total matches by industry
- Score distribution
- Performance overview

### `generate_excel_report(df, filename='industry_analysis_detailed.xlsx')`
Generates a detailed Excel report with multiple sheets:
- Industry Overview
- Keyword Analysis
- Score Analysis

### `main(data_path)`
Main function that:
1. Reads and processes the data
2. Cleans numeric columns
3. Calculates total matches and overall scores
4. Creates visualizations
5. Generates Excel report
6. Prints summary statistics

## Metrics Description
- `n_matched_keyword_<culture_name>`: Count of words related to specific cultural domain seed words
- `pct_matched_keyword_<culture_name>`: Percentage of culture-related words in the text
- `avg_score_on_keyword_<culture_name>`: Average similarity score between text words and cultural domain seed words
- `avg_score_on_matched_keyword_<culture_name>`: Average similarity score for matched words only

## Key Findings
- Total Industries Analyzed: 37
- Most Represented Industry: Property Development

### Top 3 Industries by Average Matches
1. Resources: 83,970 matches
2. Industry: 68,038 matches
3. Banking: 67,967 matches

### Top 3 Industries by Average Score
1. Education: 10.52
2. Financial Services: 10.19
3. Paper & Printing Materials: 10.11

## Usage
1. Ensure all dependencies are installed
2. Prepare input data:
   - Tokenize text using pythainlp
   - Generate word counts
   - Set up seed words for cultural values
3. Run the word embedding analysis
4. Generate visualizations and reports

## Notes
- All numeric values are cleaned to remove commas and percentage signs
- Scores are averaged across categories for overall performance metrics
- Thai language specific processing using pythainlp
- Word embeddings based on Thai2fit pre-trained models
- Visualizations include interactive elements and detailed annotations