import feedparser
from datetime import date, datetime
from transformers import pipeline
from typing import Literal, Optional, Union


class Sentiment(object):
    YAHOO_FEED = 'https://finance.yahoo.com/rss/headline?s={ticker}'

    def __init__(self, model: str = 'ProsusAI/finbert'):
        self.model = model
        self.pipe = pipeline(task='text-classification', model=self.model)

    @staticmethod
    def _has_keywords_and_date(keywords: list, feed_entry, keyword_logic: Literal['any', 'all', 'majority'] = 'any', min_date: date = None):
        if min_date is not None:
            feed_date = date(
                year=feed_entry.published_parsed.tm_year,
                month=feed_entry.published_parsed.tm_mon,
                day=feed_entry.published_parsed.tm_mday
            )
            if feed_date < min_date:
                return False

        text_low = feed_entry.summary.lower()
        if keyword_logic == 'all':
            has_keywords = True
            for keyword in keywords:
                if keyword.lower() not in text_low:
                    has_keywords = False
                    break
            return has_keywords
        elif keyword_logic == 'any':
            has_keywords = False
            for keyword in keywords:
                if keyword.lower() in text_low:
                    has_keywords = True
                    break
            return has_keywords
        else:
            len_keywords = len(keywords)
            has_count = 0
            for keyword in keywords:
                if keyword.lower() in text_low:
                    has_count += 1
            return has_count / len_keywords >= 0.50

    def _run_sentiment(
            self,
            ticker: str,
            keywords: Union[str, list[str]],
            feed_url: str = YAHOO_FEED,
            keyword_logic: Literal['any', 'all', 'majority'] = 'any',
            print_found: bool = False,
            min_date: Optional[date] = None
    ):
        formatted_url = feed_url.format(ticker=ticker)
        feed = feedparser.parse(formatted_url)

        if isinstance(keywords, str):
            keywords = [keywords]
            keyword_logic = 'all'

        total_score = 0
        number_articles = 0
        for entry in feed.entries:
            if self._has_keywords_and_date(keywords=keywords, feed_entry=entry, keyword_logic=keyword_logic, min_date=min_date):
                sentiment = self.pipe(entry.summary)[0]
                if sentiment['label'] == 'positive':
                    total_score += sentiment['score']
                    number_articles += 1
                elif sentiment['label'] == 'negative':
                    total_score -= sentiment['score']
                    number_articles += 1
                if print_found:
                    print(f'Title:\t\t{entry.title}')
                    print(f'Link:\t\t{entry.link}')
                    print(f'Published:\t{entry.published}')
                    print(f'Summary:\t{entry.summary}')
                    print(f'Sentiment:\tScore: {sentiment["score"]:,.5f} - Label: {sentiment["label"]}')
                    print(('*' * 150) + '\n')

        if number_articles == 0:
            return {'label': 'no articles found', 'score': None}

        final_score = total_score / number_articles
        if final_score >= 0.15:
            label = 'positive'
        elif final_score <= -0.15:
            label = 'negative'
        else:
            label = 'neutral'
        return {'label': label, 'score': final_score, 'matching_articles': number_articles}

    def get_sentiment_lately(
            self,
            ticker: str,
            keywords: Union[str, list[str]],
            feed_url: str = YAHOO_FEED,
            keyword_logic: Literal['any', 'all', 'majority'] = 'any',
            print_found: bool = False
    ) -> dict:
        return self._run_sentiment(
            ticker=ticker,
            keywords=keywords,
            feed_url=feed_url,
            keyword_logic=keyword_logic,
            print_found=print_found,
            min_date=None
        )

    def get_sentiment_from_date(
            self,
            ticker: str,
            keywords: Union[str, list[str]],
            min_date: date,
            feed_url: str = YAHOO_FEED,
            keyword_logic: Literal['any', 'all', 'majority'] = 'any',
            print_found: bool = False,
    ) -> dict:
        return self._run_sentiment(
            ticker=ticker,
            keywords=keywords,
            feed_url=feed_url,
            keyword_logic=keyword_logic,
            print_found=print_found,
            min_date=min_date
        )


if __name__ == '__main__':
    sent = Sentiment()
    # alk_value = sent.get_sentiment_lately(
    #     ticker='ALK',
    #     keywords=['alaska', 'airlines', 'air'],
    #     keyword_logic='majority',
    #     print_found=True
    # )
    # print(f'{alk_value=}')
    #
    # print('\n' + ('#' * 150) + '\n')
    # alk_value = sent.get_sentiment_from_date(
    #     ticker='ALK',
    #     keywords=['alaska', 'airlines', 'air'],
    #     min_date=date(2024, 8, 21),
    #     keyword_logic='majority',
    #     print_found=True
    # )
    # print(f'{alk_value=}')

    t = sent.pipe('Alaska Airlines is going downhill fast after CEO is fired and the profits are down -80% in the last month, looks like Alaska will be insolvent before the new year, do not buy this stock if you like your money')
    print(t)
    t = sent.pipe('Alaska Airlines is going downhill fast after CEO is fired and the profits are down -80% in the last month, looks like Alaska will be insolvent before the new year, do not buy this stock if you like your money')
    print(t)
    t = sent.pipe('Alaska Airlines is going downhill fast after CEO is fired and the profits are down -80% in the last month, looks like Alaska will be insolvent before the new year, do not buy this stock if you like your money')
    print(t)
    t = sent.pipe('Alaska Airlines is going downhill fast after CEO is fired and the profits are down -80% in the last month, looks like Alaska will be insolvent before the new year, do not buy this stock if you like your money')
    print(t)










