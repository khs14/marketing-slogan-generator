from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import cmudict, wordnet
from nltk.tokenize import word_tokenize
import streamlit as st
import nltk
from dataclasses import dataclass
from typing import List, Dict, Set
import numpy as np
from collections import defaultdict
import re
import random
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


nltk.download('punkt')
nltk.download('cmudict')
nltk.download('vader_lexicon')
nltk.download('wordnet')


@dataclass
class SloganMetrics:
    """Enhanced metrics for slogan evaluation"""
    memorability: float
    emotional_impact: float
    brand_relevance: float
    originality: float
    rhythm_score: float
    clarity: float
    overall_score: float


class RhymeAnalyzer:
    """Handles rhyme detection and analysis"""

    def __init__(self):
        self.pronouncing_dict = cmudict.dict()

    def get_rhymes(self, word: str) -> Set[str]:
        """Get rhyming words for a given word"""
        word = word.lower()
        if word not in self.pronouncing_dict:
            return set()

        def get_rhyme_pattern(pronunciations):
            return tuple(pronunciations[pronunciations.index(max(p for p in pronunciations if p[-1].isdigit())):])

        word_pronunciations = self.pronouncing_dict[word][0]
        rhyme_pattern = get_rhyme_pattern(word_pronunciations)

        rhymes = set()
        for dict_word in self.pronouncing_dict:
            if dict_word == word:
                continue
            for pronunciation in self.pronouncing_dict[dict_word]:
                if len(pronunciation) >= len(rhyme_pattern) and pronunciation[-len(rhyme_pattern):] == rhyme_pattern:
                    rhymes.add(dict_word)
                    break
        return rhymes


class EnhancedSloganGenerator:
    def __init__(self):
        """Initialize the enhanced slogan generator"""
        self.rhyme_analyzer = RhymeAnalyzer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Load and organize word banks
        self.load_word_banks()
        self.load_patterns()

        # Initialize cache for performance
        self.rhyme_cache = {}
        self.syllable_cache = {}

    def load_word_banks(self):
        """Load comprehensive word banks for slogan generation"""
        self.power_words = {
            'innovation': [
                'transform', 'innovate', 'revolutionize', 'pioneer', 'breakthrough',
                'cutting-edge', 'next-gen', 'future-proof', 'advanced', 'smart'
            ],
            'quality': [
                'premium', 'excellence', 'finest', 'superior', 'masterful',
                'exceptional', 'ultimate', 'pristine', 'flawless', 'perfect'
            ],
            'emotional': [
                'love', 'dream', 'passion', 'heart', 'joy', 'delight',
                'inspire', 'embrace', 'cherish', 'celebrate'
            ],
            'trust': [
                'trusted', 'reliable', 'proven', 'guaranteed', 'authentic',
                'certified', 'secure', 'dependable', 'steadfast', 'honest'
            ],
            'urgency': [
                'now', 'today', 'instant', 'immediate', 'limited',
                'exclusive', 'special', 'unique', 'rare', 'essential'
            ]
        }

        self.industry_patterns = {
            'tech': {
                'keywords': ['digital', 'smart', 'connected', 'innovative', 'tech'],
                'phrases': [
                    "Tomorrow's technology today",
                    "Innovation meets simplicity",
                    "Powering your digital future"
                ]
            },
            'food': {
                'keywords': ['taste', 'fresh', 'delicious', 'flavor', 'culinary'],
                'phrases': [
                    "Taste the difference",
                    "Fresh from nature",
                    "Culinary excellence"
                ]
            },
            'fashion': {
                'keywords': ['style', 'elegant', 'trendy', 'fashion', 'luxurious'],
                'phrases': [
                    "Style redefined",
                    "Fashion forward",
                    "Elegant by design"
                ]
            },
            'health': {
                'keywords': ['wellness', 'healthy', 'natural', 'vitality', 'fitness'],
                'phrases': [
                    "Your wellness journey",
                    "Natural healing power",
                    "Health reimagined"
                ]
            }
        }

    def load_patterns(self):
        """Load advanced slogan patterns with dynamic elements"""
        self.base_patterns = [
            "{brand} - {benefit} {audience} {value}",
            "The {quality} way to {action}",
            "{value} meets {benefit}",
            "Your {benefit} {solution}",
            "Where {value} meets {excellence}",
            "{action} with {brand}",
            "The {industry} leader in {value}",
            "{brand}: {benefit} {redefined}",
            "Experience {value} {differently}",
            "{transform} your {world} with {brand}"
        ]

        self.pattern_modifiers = {
            'prefix': ['Simply', 'Just', 'Pure', 'Beyond', 'Always'],
            'suffix': ['and more', 'reimagined', 'redefined', 'evolved'],
            'connector': ['for', 'with', 'through', 'by'],
        }

    def generate_pattern_variations(self, pattern: str) -> List[str]:
        """Generate variations of a base pattern"""
        variations = [pattern]

        # Add prefix variations
        variations.extend(
            [f"{prefix} {pattern}" for prefix in self.pattern_modifiers['prefix']])

        # Add suffix variations
        variations.extend(
            [f"{pattern}, {suffix}" for suffix in self.pattern_modifiers['suffix']])

        return variations

    def count_syllables(self, word: str) -> int:
        """Enhanced syllable counting with caching"""
        if word in self.syllable_cache:
            return self.syllable_cache[word]

        word = word.lower()
        if word in self.rhyme_analyzer.pronouncing_dict:
            count = len([ph for ph in self.rhyme_analyzer.pronouncing_dict[word][0]
                        if ph[-1].isdigit()])
        else:
            # Fallback syllable counting algorithm
            count = len(re.findall(r'[aeiou]+', word, re.I))
            count -= len(re.findall(r'[aeiou]+[aeiou]+', word, re.I))
            count -= len(re.findall(r'e$', word, re.I))
            count = max(1, count)

        self.syllable_cache[word] = count
        return count

    def calculate_rhythm_score(self, words: List[str]) -> float:
        """Calculate rhythmic quality of the slogan"""
        if not words:
            return 0.0

        syllables = [self.count_syllables(word) for word in words]

        # Check for alternating pattern
        alternating = sum(abs(syllables[i] - syllables[i-1]) == 1
                          for i in range(1, len(syllables)))

        # Check for balanced structure
        total_syllables = sum(syllables)
        balance = 1.0 - abs(total_syllables / len(syllables) - 3) / 3

        return (alternating / (len(words) - 1) * 0.6 + balance * 0.4)

    def evaluate_slogan(self, slogan: str, brand_context: Dict) -> SloganMetrics:
        """Comprehensive slogan evaluation"""
        words = word_tokenize(slogan.lower())

        # Calculate memorability
        rhyme_score = 1.0 if any(word in self.rhyme_analyzer.get_rhymes(words[0])
                                 for word in words[1:]) else 0.0
        alliteration_score = self.calculate_alliteration_score(words)
        memorability = (rhyme_score * 0.4 + alliteration_score * 0.6)

        # Calculate emotional impact
        sentiment_scores = self.sentiment_analyzer.polarity_scores(slogan)
        emotional_words = sum(1 for word in words
                              if any(word in word_list
                                     for word_list in self.power_words.values()))
        emotional_impact = (sentiment_scores['compound'] + 1) / 2 * 0.6 + \
            (emotional_words / len(words)) * 0.4

        # Calculate brand relevance
        industry_words = self.industry_patterns[brand_context['industry']]['keywords']
        relevant_words = sum(1 for word in words
                             if word in industry_words
                             or word in brand_context['values'])
        brand_relevance = min(1.0, relevant_words / len(words) * 1.5)

        # Calculate originality
        cliches = {'best', 'better', 'quality',
                   'excellence', 'leader', 'leading'}
        originality = 1.0 - \
            sum(1 for word in words if word in cliches) / len(words)

        # Calculate rhythm score
        rhythm_score = self.calculate_rhythm_score(words)

        # Calculate clarity
        avg_word_length = sum(len(word) for word in words) / len(words)
        clarity = max(0.0, 1.0 - (avg_word_length - 5) * 0.1)

        # Calculate overall score with weighted components
        weights = {
            'memorability': 0.25,
            'emotional_impact': 0.20,
            'brand_relevance': 0.25,
            'originality': 0.15,
            'rhythm': 0.10,
            'clarity': 0.05
        }

        overall_score = sum([
            memorability * weights['memorability'],
            emotional_impact * weights['emotional_impact'],
            brand_relevance * weights['brand_relevance'],
            originality * weights['originality'],
            rhythm_score * weights['rhythm'],
            clarity * weights['clarity']
        ])

        return SloganMetrics(
            memorability=memorability,
            emotional_impact=emotional_impact,
            brand_relevance=brand_relevance,
            originality=originality,
            rhythm_score=rhythm_score,
            clarity=clarity,
            overall_score=overall_score
        )

    def calculate_alliteration_score(self, words: List[str]) -> float:
        """Calculate alliteration score"""
        if not words:
            return 0.0

        first_letters = [word[0] for word in words if word]
        consecutive_matches = sum(1 for i in range(len(first_letters) - 1)
                                  if first_letters[i] == first_letters[i + 1])

        return min(1.0, consecutive_matches / (len(words) - 1))

    def generate_slogans(self, brand_context: Dict, num_slogans: int = 5) -> List[Dict]:
        """Generate and evaluate multiple slogans"""
        generated_slogans = []
        industry = brand_context['industry']

        for _ in range(num_slogans * 3):  # Generate extra to filter best ones
            # Select and modify pattern
            base_pattern = random.choice(self.base_patterns)
            pattern = random.choice(
                self.generate_pattern_variations(base_pattern))

            # Get industry-specific elements
            industry_keywords = self.industry_patterns[industry]['keywords']
            industry_phrases = self.industry_patterns[industry]['phrases']

            # Fill in the pattern with context-aware substitutions
            slogan = pattern.format(
                brand=brand_context['name'],
                benefit=random.choice(industry_keywords +
                                      self.power_words['innovation'] +
                                      self.power_words['quality']),
                quality=random.choice(self.power_words['quality']),
                action=random.choice(self.power_words['innovation']),
                value=random.choice(brand_context['values'] +
                                    self.power_words['emotional']),
                audience=brand_context['target_audience'],
                industry=industry,
                solution=random.choice(
                    ['solution', 'choice', 'partner', 'advantage']),
                excellence=random.choice(self.power_words['quality']),
                transform=random.choice(self.power_words['innovation']),
                world=random.choice(
                    [industry, 'future', 'possibilities', 'potential']),
                redefined=random.choice(self.pattern_modifiers['suffix']),
                differently=random.choice(
                    ['differently', 'better', 'uniquely'])
            )

            # Sometimes use industry-specific phrases
            if random.random() < 0.3:
                slogan = random.choice(industry_phrases)

            # Evaluate the slogan
            metrics = self.evaluate_slogan(slogan, brand_context)

            generated_slogans.append({
                'slogan': slogan,
                'metrics': metrics
            })

        # Sort by overall score and remove duplicates
        sorted_slogans = sorted(
            generated_slogans,
            key=lambda x: x['metrics'].overall_score,
            reverse=True
        )

        # Remove duplicates while preserving order
        unique_slogans = []
        seen = set()
        for slogan in sorted_slogans:
            if slogan['slogan'] not in seen:
                unique_slogans.append(slogan)
                seen.add(slogan['slogan'])
                if len(unique_slogans) == num_slogans:
                    break

        return unique_slogans


def get_tone_patterns():
    """Get patterns and modifiers based on tone"""
    return {
        'Professional': {
            'modifiers': ['Innovative', 'Strategic', 'Excellence in', 'Leading'],
            'patterns': [
                "{brand}: Excellence in {value}",
                "Professional {benefit} solutions",
                "Leading the way in {industry}"
            ],
            'keywords': ['excellence', 'professional', 'innovative', 'leading']
        },
        'Friendly': {
            'modifiers': ['Simply', 'Happily', 'Naturally', 'Together'],
            'patterns': [
                "Your friendly {benefit} partner",
                "Making {value} fun",
                "Together in {value}"
            ],
            'keywords': ['together', 'friendly', 'happy', 'fun']
        },
        'Inspirational': {
            'modifiers': ['Dream', 'Imagine', 'Believe', 'Achieve'],
            'patterns': [
                "Dream big with {brand}",
                "Imagine the possibilities of {value}",
                "Your journey to {benefit}"
            ],
            'keywords': ['dream', 'imagine', 'believe', 'inspire']
        },
        'Luxurious': {
            'modifiers': ['Premium', 'Exclusive', 'Elegant', 'Refined'],
            'patterns': [
                "Experience exceptional {value}",
                "The art of {benefit}",
                "Luxury redefined"
            ],
            'keywords': ['luxury', 'premium', 'exclusive', 'elegant']
        },
        'Casual': {
            'modifiers': ['Just', 'Simply', 'Naturally', 'Easy'],
            'patterns': [
                "Simply {benefit}",
                "Just what you need",
                "Easy {value} for everyone"
            ],
            'keywords': ['simple', 'easy', 'natural', 'casual']
        },
        'Bold': {
            'modifiers': ['Revolutionize', 'Transform', 'Dominate', 'Power'],
            'patterns': [
                "Revolutionize your {industry}",
                "Transform your {value}",
                "Unleash the power of {benefit}"
            ],
            'keywords': ['revolutionary', 'powerful', 'bold', 'dynamic']
        }
    }


def create_metrics_sunburst(metrics: SloganMetrics):
    """Create a sunburst chart for metrics visualization"""
    metrics_dict = {k: v for k, v in metrics.__dict__.items()
                    if k != 'overall_score'}

    # Create hierarchical data for sunburst
    data = {
        'names': ['Metrics'] + list(metrics_dict.keys()),
        'parents': [''] + ['Metrics'] * len(metrics_dict),
        'values': [sum(metrics_dict.values())] + list(metrics_dict.values()),
        'textinfo': 'label+value',
        'hovertemplate': '%{label}<br>Score: %{value:.2f}<extra></extra>'
    }

    fig = go.Figure(go.Sunburst(
        labels=data['names'],
        parents=data['parents'],
        values=data['values'],
        branchvalues='total',
    ))

    fig.update_layout(
        title="Metrics Distribution",
        width=400,
        height=400
    )

    return fig


def create_comparison_chart(slogans: List[Dict]):
    """Create a comparison chart for all generated slogans"""
    # Prepare data for visualization
    df_data = []
    for i, slogan in enumerate(slogans, 1):
        metrics = slogan['metrics'].__dict__
        for metric, value in metrics.items():
            if metric != 'overall_score':
                df_data.append({
                    'Slogan Number': f'Slogan {i}',
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': value
                })

    df = pd.DataFrame(df_data)

    fig = px.bar(df,
                 x='Slogan Number',
                 y='Score',
                 color='Metric',
                 barmode='group',
                 title='Comparison of Metrics Across Slogans')

    fig.update_layout(
        height=500,
        xaxis_title="Slogans",
        yaxis_title="Score",
        legend_title="Metrics"
    )

    return fig


def main():
    st.set_page_config(layout="wide")
    st.title("Slogan Generator")

    # Initialize slogan generator
    generator = EnhancedSloganGenerator()
    tone_patterns = get_tone_patterns()

    # Input form with tone selection
    st.header("Brand Information")
    with st.form("brand_info"):
        col1, col2 = st.columns(2)

        with col1:
            brand_name = st.text_input("Brand Name")
            industry = st.selectbox("Industry",
                                    options=['tech', 'food', 'fashion', 'health'])
            values = st.multiselect("Brand Values",
                                    options=['innovation', 'simplicity', 'reliability',
                                             'quality', 'sustainability', 'luxury',
                                             'affordability', 'creativity'])

        with col2:
            st.subheader("Tone Selection")

            # Primary tone
            primary_tone = st.selectbox(
                "Primary Tone",
                options=list(tone_patterns.keys()),
                help="The main communication style for your slogan"
            )

            # Secondary tone (optional)
            secondary_tone = st.selectbox(
                "Secondary Tone (Optional)",
                options=['None'] + list(tone_patterns.keys()),
                help="Add a secondary tone to create a unique voice"
            )

            # Tone strength
            tone_strength = st.slider(
                "Tone Strength",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="How strongly to apply the selected tone(s)"
            )

            # Add specific emphasis
            emphasis = st.multiselect(
                "Additional Emphasis",
                options=[
                    'Call to Action',
                    'Question Format',
                    'Emotional Appeal',
                    'Benefit Focused',
                    'Problem-Solution'
                ],
                help="Add specific elements to your slogan"
            )

        num_slogans = st.slider("Number of Slogans to Generate", 3, 10, 5)
        submit_button = st.form_submit_button("Generate Slogans")

    if submit_button and brand_name and industry and values:
        # Construct tone-based patterns and modifiers
        selected_patterns = tone_patterns[primary_tone]['patterns']
        selected_modifiers = tone_patterns[primary_tone]['modifiers']

        if secondary_tone != 'None':
            selected_patterns.extend(tone_patterns[secondary_tone]['patterns'])
            selected_modifiers.extend(
                tone_patterns[secondary_tone]['modifiers'])

        # Modify generator patterns based on tone
        generator.base_patterns = selected_patterns
        generator.pattern_modifiers['prefix'] = selected_modifiers

        # Add emphasis patterns if selected
        if 'Question Format' in emphasis:
            generator.base_patterns.extend([
                "Why choose anything but {brand}?",
                "Ready for {benefit}?",
                "Isn't it time for {value}?"
            ])
        if 'Call to Action' in emphasis:
            generator.base_patterns.extend([
                "Discover {value} today",
                "Experience {benefit} now",
                "Join the {industry} revolution"
            ])
        if 'Problem-Solution' in emphasis:
            generator.base_patterns.extend([
                "No more {problem}. Just {solution}",
                "Transform your {problem} into {benefit}",
                "Finally, a {solution} that works"
            ])

        brand_context = {
            'name': brand_name,
            'industry': industry,
            'values': values,
            'target_audience': primary_tone + ("-" + secondary_tone if secondary_tone != 'None' else "")
        }
        with st.spinner("Generating slogans..."):
            slogans = generator.generate_slogans(brand_context, num_slogans)

        # Create tabs for different views
        tab1, tab2 = st.tabs(["Individual Slogans", "Comparative Analysis"])

        with tab1:
            # Individual slogan analysis
            for i, result in enumerate(slogans, 1):
                with st.expander(f"Slogan {i}: {result['slogan']}"):
                    metrics = result['metrics']

                    # Metrics visualization
                    col1, col2 = st.columns(2)

                    with col1:
                        st.plotly_chart(create_metrics_sunburst(metrics))

                    with col2:
                        # Radar chart
                        metrics_dict = {k: v for k, v in metrics.__dict__.items()
                                        if k != 'overall_score'}

                        fig = go.Figure(data=go.Scatterpolar(
                            r=list(metrics_dict.values()),
                            theta=list(map(lambda x: x.replace('_', ' ').title(),
                                           metrics_dict.keys())),
                            fill='toself'
                        ))

                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )),
                            showlegend=False,
                            title="Metrics Radar Chart"
                        )

                        st.plotly_chart(fig)

                    st.markdown(
                        f"**Overall Score:** {metrics.overall_score:.2f}")

        with tab2:
            # Comparative analysis
            st.header("Comparative Analysis")

            # Overall comparison chart
            st.plotly_chart(create_comparison_chart(
                slogans), use_container_width=True)

            # Top performers
            st.subheader("Top Performing Slogans")
            top_metrics = ['memorability',
                           'emotional_impact', 'brand_relevance']

            for metric in top_metrics:
                best_slogan = max(slogans,
                                  key=lambda x: getattr(x['metrics'], metric))
                st.markdown(f"**Best {metric.replace('_', ' ').title()}:** "
                            f"{best_slogan['slogan']} "
                            f"(Score: {getattr(best_slogan['metrics'], metric):.2f})")

    else:
        if submit_button:
            st.warning("Please fill in all required fields.")


if __name__ == "__main__":
    main()
