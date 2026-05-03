import json
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


class YoutubeScraper:
	"""
	Production-ready youtube scraper with logging, rate limiting, and error recovery.
	"""

	BASE_HEADERS = {'User-Agent': 'Mozilla/5.0 (Educational Purpose) YoutubeScraper/1.0'}
	BASE_URL = 'https://www.youtube.com/watch?v='

	VERIFIED_LABELS = {'Verified'}
	VERIFIED_STYLES = {'BADGE_STYLE_TYPE_VERIFIED'}
	VERIFIED_ICONS = {'CHECK_CIRCLE_THICK'}

	MAX_WORKERS = 10  # concurrent threads (start here; raise if stable)
	SAVE_INTERVAL = 50  # save progress every N completions

	def __init__(self, output_dir='scraped_data', logger=None, session=None):
		self.logger = logger
		self.session = session

		self.rate_limit = 10_000
		self.time_window = 60
		self.request_times = deque()

		self._rate_lock = threading.Lock()
		self._progress_lock = threading.Lock()

		self._robots_cache: dict[str, bool] = {}
		self._robots_lock = threading.Lock()

		self.output_dir = output_dir
		os.makedirs(self.output_dir, exist_ok=True)

	# ------------------------------------------------------------------
	# Public entry point
	# ------------------------------------------------------------------

	def scrape_videos(self, video_ids: list[str]) -> pd.DataFrame:
		"""
		Main entry point. Accept a list of video IDs and return a DataFrame of new features.
		"""
		rows = self.load_progress()
		scraped_ids = {row['video_id'] for row in rows}
		pending = [vid for vid in video_ids if vid not in scraped_ids]

		if scraped_ids:
			self.logger.info(f'Resuming: {len(scraped_ids)} already done, {len(pending)} remaining.')

		unsaved_buf: list[dict] = []

		with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
			future_to_id = {pool.submit(self._scrape_one_safe, vid): vid for vid in pending}

			with tqdm(total=len(pending)) as pbar:
				for future in as_completed(future_to_id):
					_ = future_to_id[future]
					result = future.result()
					pbar.update(1)

					if result is None:
						continue

					with self._progress_lock:
						rows.append(result)
						unsaved_buf.append(result)

						if len(unsaved_buf) >= self.SAVE_INTERVAL:
							self.save_progress(unsaved_buf, 'progress.txt')
							unsaved_buf = []

		if unsaved_buf:
			self.save_progress(unsaved_buf, 'progress.txt')

		df = pd.DataFrame(rows).drop_duplicates(subset=['video_id'])
		self.export_data(df, 'videos')
		self.logger.info(f'Total scraped: {len(df)} videos')
		return df

	# ------------------------------------------------------------------
	# Per-video wrapper (catches & logs exceptions so pool keeps going)
	# ------------------------------------------------------------------

	def _scrape_one_safe(self, vid_id: str) -> dict | None:
		url = self.BASE_URL + vid_id
		self.logger.info(f'Scraping video: {vid_id}')

		if not self.check_robots_txt(url):
			self.logger.warning(f'Blocked by robots.txt — skipping {vid_id}')
			return None

		try:
			row = self._scrape_single(url, vid_id)
			self.logger.info(f'Successfully scraped {vid_id}')
			return row
		except Exception as e:
			self.logger.error(f'Failed to scrape {vid_id}: {e}')
			return None

	# ------------------------------------------------------------------
	# Core scraping logic (unchanged from original)
	# ------------------------------------------------------------------

	def _scrape_single(self, url: str, video_id: str) -> dict:
		soup = self._fetch_page(url)
		player_data = self._extract_json(soup, 'ytInitialPlayerResponse')
		initial_data = self._extract_json(soup, 'ytInitialData')

		chapters = self.get_chapters(player_data, initial_data)
		playability = self._extract_playability(player_data)
		cards = self._extract_cards(player_data)
		verified = self._extract_verified(initial_data)
		comments_disabled = self._extract_comments_disabled(initial_data)
		has_paid_promotion = self._extract_paid_promotion(player_data)

		return {
			'video_id': video_id,
			'chapter_count': len(chapters),
			'chapters': chapters,
			'playability_status': playability['status'],
			'supports_miniplayer': playability['supports_miniplayer'],
			'card_count': cards['card_count'],
			'cards': cards['card_items'],
			'is_verified': verified['is_verified'],
			'badge_labels': verified['badge_labels'],
			'comments_disabled': comments_disabled,
			'has_paid_promotion': has_paid_promotion,
		}

	def _fetch_page(self, url: str) -> BeautifulSoup:
		self.enforce_rate_limit()
		response = self.session.get(url)
		response.raise_for_status()
		return BeautifulSoup(response.text, 'html.parser')

	# ------------------------------------------------------------------
	# Chapter extraction (unchanged)
	# ------------------------------------------------------------------

	def get_chapters(self, player_data, initial_data):
		return (
			self._chapters_from_player_data(player_data)
			or self._chapters_from_initial_data(initial_data)
			or self._chapters_from_description(player_data)
			or []
		)

	def _chapters_from_player_data(self, player_data):
		try:
			return self._parse_markers(self._get_markers_map(player_data))
		except (KeyError, TypeError):
			return []

	def _chapters_from_initial_data(self, initial_data):
		try:
			return self._parse_markers(self._get_markers_map(initial_data))
		except (KeyError, TypeError):
			return []

	def _chapters_from_description(self, player_data):
		try:
			description = player_data['videoDetails']['shortDescription']
			matches = re.findall(r'(\d{1,2}:\d{2})\s+(.+)', description)
			return [{'title': t.strip(), 'start_seconds': self._timestamp_to_seconds(ts)} for ts, t in matches]
		except (KeyError, TypeError):
			return []

	# ------------------------------------------------------------------
	# Feature extractors (unchanged)
	# ------------------------------------------------------------------

	def _extract_playability(self, player_data):
		defaults = {'status': 'UNKNOWN', 'supports_miniplayer': False}
		if not player_data:
			return defaults
		try:
			playability = player_data['playabilityStatus']
			status = playability.get('status', 'UNKNOWN')
			supports_miniplayer = (
				player_data.get('microformat', {}).get('playerMicroformatRenderer', {}).get('isFamilySafe', False)
			)
			return {'status': status, 'supports_miniplayer': supports_miniplayer}
		except (KeyError, TypeError):
			return defaults

	def _extract_cards(self, player_data):
		if not player_data:
			return {'card_count': 0, 'card_items': []}
		try:
			cards = player_data['cards']['cardCollectionRenderer']['cards']
			card_items = [
				{
					'teaser_text': (
						c['cardRenderer']
						.get('teaser', {})
						.get('simpleCardTeaserRenderer', {})
						.get('message', {})
						.get('simpleText', '')
					),
					'start_ms': c['cardRenderer'].get('startCardActiveMs'),
				}
				for c in cards
			]
			return {'card_count': len(card_items), 'card_items': card_items}
		except (KeyError, TypeError):
			return {'card_count': 0, 'card_items': []}

	def _extract_verified(self, initial_data):
		defaults = {'is_verified': False, 'badge_labels': []}
		if not initial_data:
			return defaults
		try:
			badges = initial_data['contents']['twoColumnWatchNextResults']['results']['results']['contents'][1][
				'videoSecondaryInfoRenderer'
			]['owner']['videoOwnerRenderer']['badges']
			is_verified, badge_labels = False, []
			for b in badges:
				renderer = b.get('metadataBadgeRenderer', {})
				label = renderer.get('accessibilityData', {}).get('label', '')
				style = renderer.get('style', '')
				icon_type = renderer.get('icon', {}).get('iconType', '')
				badge_labels.append(label)
				if (
					label in self.VERIFIED_LABELS
					or any(style.startswith(s) for s in self.VERIFIED_STYLES)
					or icon_type in self.VERIFIED_ICONS
				):
					is_verified = True
			return {'is_verified': is_verified, 'badge_labels': badge_labels}
		except (KeyError, TypeError):
			return defaults

	def _extract_comments_disabled(self, initial_data):
		try:
			contents = initial_data['contents']['twoColumnWatchNextResults']['results']['results']['contents']
			for item in contents:
				for c in item.get('itemSectionRenderer', {}).get('contents', []):
					if 'messageRenderer' in c:
						return True
		except (KeyError, TypeError):
			pass
		return False

	def _extract_paid_promotion(self, player_data):
		try:
			return bool(player_data['paidContentOverlay'])
		except (KeyError, TypeError):
			return False

	def _extract_json(self, soup, key):
		for script in soup.find_all('script'):
			if key not in script.text:
				continue
			match = re.search(rf'{key}\s*=\s*({{.*}});', script.text)
			if match:
				try:
					return json.loads(match.group(1))
				except json.JSONDecodeError:
					pass
		return None

	def _get_markers_map(self, data):
		return data['playerOverlays']['playerOverlayRenderer']['decoratedPlayerBarRenderer'][
			'decoratedPlayerBarRenderer'
		]['playerBar']['multiMarkersPlayerBarRenderer']['markersMap']

	def _parse_markers(self, markers):
		for item in markers:
			if item.get('key') == 'AUTO_CHAPTERS':
				return [
					{
						'title': ch['chapterRenderer']['title']['simpleText'],
						'start_seconds': ch['chapterRenderer']['timeRangeStartMillis'] // 1000,
					}
					for ch in item['value']['chapters']
				]
		return []

	def _timestamp_to_seconds(self, time_str):
		minutes, seconds = map(int, time_str.split(':'))
		return minutes * 60 + seconds

	# ------------------------------------------------------------------
	# Rate limiting  ✅ FIX 3: thread-safe via _rate_lock
	# ------------------------------------------------------------------

	def enforce_rate_limit(self):
		with self._rate_lock:
			now = time.time()
			while self.request_times and now - self.request_times[0] > self.time_window:
				self.request_times.popleft()

			if len(self.request_times) >= self.rate_limit:
				wait_time = self.time_window - (now - self.request_times[0]) + 1
				self.logger.info(f'Rate limit reached. Waiting {wait_time:.2f}s...')
				time.sleep(wait_time)

			self.request_times.append(time.time())

	# ------------------------------------------------------------------
	# Robots.txt  ✅ FIX 4: cache per domain
	# ------------------------------------------------------------------

	def check_robots_txt(self, url: str) -> bool:
		parsed = urlparse(url)
		domain = f'{parsed.scheme}://{parsed.netloc}'

		# return cached result if we've checked this domain before
		with self._robots_lock:
			if domain in self._robots_cache:
				return self._robots_cache[domain]

		# fetch & parse outside the lock (slow I/O should not block other threads)
		try:
			rp = RobotFileParser()
			rp.set_url(f'{domain}/robots.txt')
			rp.read()
			allowed = rp.can_fetch(self.session.headers.get('User-Agent', ''), url)
			self.logger.info(f'robots.txt {domain}: {"allowed" if allowed else "blocked"}')
		except Exception as e:
			self.logger.warning(f'Could not read robots.txt for {domain}: {e}')
			allowed = False

		with self._robots_lock:
			self._robots_cache[domain] = allowed

		return allowed

	# ------------------------------------------------------------------
	# Persistence (unchanged logic, but callers hold _progress_lock)
	# ------------------------------------------------------------------

	def save_progress(self, videos, filename='progress.txt'):
		try:
			filepath = os.path.join(self.output_dir, filename)
			with open(filepath, 'a') as f:
				for video in videos:
					f.write(json.dumps(video) + '\n')
			self.logger.info(f'Progress appended ({len(videos)} rows) to {filepath}')
		except Exception as e:
			self.logger.error(f'Failed to save progress: {e}')

	def load_progress(self, filename='progress.txt') -> list[dict]:
		filepath = os.path.join(self.output_dir, filename)
		if not os.path.exists(filepath):
			return []

		seen = {}
		with open(filepath) as f:
			for line in f:
				line = line.strip()
				if line:
					try:
						row = json.loads(line)
						seen[row['video_id']] = row
					except json.JSONDecodeError:
						pass

		return list(seen.values())

	def export_data(self, videos, base_filename='videos'):
		try:
			filepath = os.path.join(self.output_dir, base_filename)
			df = pd.DataFrame(videos)
			df.to_csv(f'{filepath}.csv', index=False, encoding='utf-8')
			self.logger.info(f'Exported to {base_filename}.csv')
		except Exception as e:
			self.logger.error(f'Failed to export data: {e}')
