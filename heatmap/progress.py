import json
import sys
import time
from tqdm import tqdm

# Machine-readable progress lines. The web app parses this prefix from stdout.
PROGRESS_PREFIX = 'HM_PROGRESS '

# Final batched rasterize is predicted as this fraction of the per-feature
# window-rasterize time measured in countRasterCells, until a finished file
# replaces it with actual_raster_seconds / count_seconds.
RASTER_TO_COUNT_RATIO = 0.25

# Initial GeoTIFF write rate, replaced by the measured pixels/second.
WRITE_PIXELS_PER_SEC = 20_000_000

# Hold the time remaining until the feature rate is past the first few samples.
# A handful of features is enough to move the percent, and too few to trust an ETA.
_ETA_MIN_FEATURES = 10
_ETA_MIN_SECONDS = 20.0

PHASE_LABEL = {
  'features': 'Processing features',
  'rasterizing': 'Rasterizing',
  'writing': 'Writing raster',
}

_EMIT_INTERVAL = 0.25
_BAR_INTERVAL = 0.1


def create_progress(total, desc, unit='it', initial=0):
  return tqdm(
    total=max(int(total), 0),
    desc=desc,
    unit=unit,
    initial=initial,
    file=sys.stderr,
    ncols=100,
    leave=False,
    smoothing=0.05,
    mininterval=0.1,
    disable=None,
  )


def progress_write(_progress, msg):
  tqdm.write(str(msg), file=sys.stdout)
  sys.stdout.flush()


class RunProgress:
  """Overall run progress, including predicted rasterize and write time.

  Percent is time already spent divided by estimated time for every file.
  Feature time uses the observed seconds per feature. Rasterize time is
  extrapolated from countRasterCells window pixels and duration, scaled by
  raster_ratio. Write time is output pixels divided by the write rate.
  Files not started yet reuse the current file's per-feature estimates.
  The displayed percent never moves backward.
  """

  def __init__(self, total_features, file_count):
    self.features_total = max(int(total_features), 0)
    self.file_count = max(int(file_count), 0)
    self.features_done = 0
    self.file_index = 0
    self.raster_ratio = RASTER_TO_COUNT_RATIO
    self.write_px_per_s = float(WRITE_PIXELS_PER_SEC)
    self.displayed_percent = 0.0

    self._completed_s = 0.0
    self._completed_features = 0
    self._completed_feature_seconds = 0.0
    self._completed_count_seconds = 0.0
    self._completed_window_px = 0
    self._template_raster_s = 0.0
    self._template_write_s = 0.0

    self._phase = 'features'
    self._file_name = ''
    self.file_features = 0
    self.file_features_done = 0
    self.file_pixels = 0
    self.feature_elapsed = 0.0
    self.count_seconds = 0.0
    self.count_window_px = 0
    self._feature_t0 = None

    self._raster_geoms_total = 0
    self._raster_geoms_done = 0
    self._raster_elapsed = 0.0
    self._raster_t0 = None

    self._write_rows_total = 0
    self._write_rows_done = 0
    self._write_elapsed = 0.0
    self._write_t0 = None

    self._last_emit = 0.0
    self._last_bar = 0.0
    self._bar = create_progress(100, 'Processing features', unit='%')

  def begin_file(self, name, n_features, width, height):
    self.file_index += 1
    self._file_name = name or ''
    self.file_features = max(int(n_features), 0)
    self.file_features_done = 0
    self.file_pixels = max(int(width), 0) * max(int(height), 0)
    self.feature_elapsed = 0.0
    self.count_seconds = 0.0
    self.count_window_px = 0
    self._feature_t0 = time.perf_counter()
    self._phase = 'features'
    self._raster_geoms_total = 0
    self._raster_geoms_done = 0
    self._raster_elapsed = 0.0
    self._raster_t0 = None
    self._write_rows_total = 0
    self._write_rows_done = 0
    self._write_elapsed = 0.0
    self._write_t0 = None
    self._refresh(force=True)

  def skip_file(self, name, n_features):
    """Count a file as finished without raster or write work."""
    self.file_index += 1
    self._file_name = name or ''
    self.features_done += max(int(n_features), 0)
    self.file_features = 0
    self.file_features_done = 0
    self._phase = 'features'
    self._refresh(force=True)

  def note_count(self, window_pixels, seconds):
    self.count_window_px += max(int(window_pixels), 0)
    self.count_seconds += max(float(seconds), 0.0)

  def update_features(self, n=1):
    self.file_features_done += n
    self.features_done += n
    if self._feature_t0 is not None:
      self.feature_elapsed = time.perf_counter() - self._feature_t0
    self._refresh()

  def end_features(self):
    self._freeze_feature_timer()
    self._refresh(force=True)

  def start_raster(self, n_geoms):
    self._freeze_feature_timer()
    self._phase = 'rasterizing'
    self._raster_geoms_total = max(int(n_geoms), 0)
    self._raster_geoms_done = 0
    self._raster_elapsed = 0.0
    self._raster_t0 = time.perf_counter()
    self._refresh(force=True)

  def update_raster(self, n=1):
    self._raster_geoms_done += n
    if self._raster_t0 is not None:
      self._raster_elapsed = time.perf_counter() - self._raster_t0
    self._refresh()

  def end_raster(self):
    self._freeze_raster_timer()
    self._refresh(force=True)

  def start_write(self, n_rows):
    self._freeze_raster_timer()
    self._phase = 'writing'
    self._write_rows_total = max(int(n_rows), 0)
    self._write_rows_done = 0
    self._write_elapsed = 0.0
    self._write_t0 = time.perf_counter()
    self._refresh(force=True)

  def update_write(self, n_rows):
    self._write_rows_done += n_rows
    if self._write_t0 is not None:
      self._write_elapsed = time.perf_counter() - self._write_t0
    self._refresh()

  def end_write(self):
    self._freeze_write_timer()
    self._refresh(force=True)

  def finish_file(self):
    """Calibrate predictions from this file, then emit while it is still current."""
    self._freeze_feature_timer()
    self._freeze_raster_timer()
    self._freeze_write_timer()
    if self.count_seconds > 0 and self._raster_geoms_total > 0:
      self.raster_ratio = self._raster_elapsed / self.count_seconds
    if self._write_elapsed > 0 and self.file_pixels > 0:
      self.write_px_per_s = self.file_pixels / self._write_elapsed
    self._refresh(force=True)
    if self.file_features_done > 0:
      self._template_raster_s = self._raster_elapsed / self.file_features_done
      self._template_write_s = self._write_elapsed / self.file_features_done
    self._completed_s += self.feature_elapsed + self._raster_elapsed + self._write_elapsed
    self._completed_feature_seconds += self.feature_elapsed
    self._completed_features += self.file_features_done
    self._completed_count_seconds += self.count_seconds
    self._completed_window_px += self.count_window_px
    self._reset_current_file()

  def close(self):
    if self._bar is None:
      return
    self._bar.close()
    self._bar = None

  def snapshot(self):
    budget = self._budget()
    eta = None
    if budget is not None:
      done, total = budget
      total = max(total, done)
      raw = 0.0 if total <= 0 else 100.0 * done / total
      raw = min(100.0, max(0.0, raw))
      self.displayed_percent = max(self.displayed_percent, raw)
      if self._eta_ready():
        eta = round(total - done, 1)
    return {
      'phase': self._phase,
      'percent': round(self.displayed_percent, 1),
      'eta_seconds': eta,
      'file': self._file_name,
      'file_index': self.file_index,
      'file_count': self.file_count,
      'features_done': self.features_done,
      'features_total': self.features_total,
    }

  def _eta_ready(self):
    """True once the remaining-time estimate is based on more than a short sample.

    Later files reuse calibrated rates. On the first file, wait through 10
    features and 20 seconds of feature work. If the file is smaller than that,
    wait until rasterizing or writing has a measured rate.
    """
    if self._completed_features > 0:
      return True
    if self.file_features_done >= _ETA_MIN_FEATURES and self.feature_elapsed >= _ETA_MIN_SECONDS:
      return True
    if self._phase == 'rasterizing' and self._raster_geoms_done > 0 and self._raster_elapsed > 0:
      return True
    if self._phase == 'writing' and self._write_rows_done > 0 and self._write_elapsed > 0:
      return True
    return False

  def _spf(self):
    n = self._completed_features + self.file_features_done
    if n <= 0:
      return None
    warmup = min(3, self.file_features) if self.file_features else 3
    still_warming = (
      self._phase == 'features'
      and self.file_features_done < warmup
      and self._completed_features < 3
    )
    if still_warming:
      return None
    return (self._completed_feature_seconds + self.feature_elapsed) / n

  def _raster_sample(self):
    """Best (window pixels, seconds, feature count) for the raster prediction."""
    if self.file_features_done >= 3 and self.count_seconds > 0 and self.count_window_px > 0:
      return self.count_window_px, self.count_seconds, self.file_features_done
    if self._completed_count_seconds > 0 and self._completed_window_px > 0 and self._completed_features > 0:
      return self._completed_window_px, self._completed_count_seconds, self._completed_features
    if self._spf() is not None and self.file_features_done > 0 and self.count_seconds > 0 and self.count_window_px > 0:
      return self.count_window_px, self.count_seconds, self.file_features_done
    return None

  def _predicted_raster_s(self, n_features):
    sample = self._raster_sample()
    if sample is None or n_features <= 0:
      return 0.0
    window_px, seconds, features = sample
    px_per_s = window_px / seconds
    if px_per_s <= 0:
      return 0.0
    expected_px = (window_px / features) * n_features
    return (expected_px / px_per_s) * self.raster_ratio

  def _pixels_write_seconds(self, pixels):
    if pixels <= 0 or self.write_px_per_s <= 0:
      return 0.0
    return pixels / self.write_px_per_s

  def _future_features(self):
    done_before = self.features_done - self.file_features_done
    accounted = done_before + self.file_features
    return max(self.features_total - accounted, 0)

  def _budget(self):
    spf = self._spf()
    if spf is None:
      return None

    feature_done = self.feature_elapsed
    if self._phase == 'features':
      remaining = max(self.file_features - self.file_features_done, 0)
      feature_total = feature_done + remaining * spf
    else:
      feature_total = feature_done

    if (
      self._phase == 'rasterizing'
      and self._raster_geoms_done > 0
      and self._raster_elapsed > 0
      and self._raster_geoms_total > 0
    ):
      raster_done = self._raster_elapsed
      raster_total = self._raster_elapsed * self._raster_geoms_total / self._raster_geoms_done
    elif self._phase == 'writing':
      raster_done = self._raster_elapsed
      raster_total = self._raster_elapsed
    else:
      raster_done = 0.0
      raster_total = self._predicted_raster_s(self.file_features)

    write_estimate = self._pixels_write_seconds(self.file_pixels)
    if (
      self._phase == 'writing'
      and self._write_rows_done > 0
      and self._write_elapsed > 0
      and self._write_rows_total > 0
    ):
      write_done = self._write_elapsed
      write_total = self._write_elapsed * self._write_rows_total / self._write_rows_done
    elif self._phase == 'writing':
      write_done = self._write_elapsed
      write_total = max(write_estimate, self._write_elapsed)
    else:
      write_done = 0.0
      write_total = write_estimate

    future = self._future_features()
    future_s = 0.0
    if future > 0:
      if self.file_features > 0:
        per_raster = raster_total / self.file_features
        per_write = write_total / self.file_features
      else:
        per_raster = self._template_raster_s
        per_write = self._template_write_s
      future_s = future * (spf + per_raster + per_write)

    done = self._completed_s + feature_done + raster_done + write_done
    total = self._completed_s + feature_total + raster_total + write_total + future_s
    return done, total

  def _freeze_feature_timer(self):
    if self._feature_t0 is None:
      return
    self.feature_elapsed = time.perf_counter() - self._feature_t0
    self._feature_t0 = None

  def _freeze_raster_timer(self):
    if self._raster_t0 is None:
      return
    self._raster_elapsed = time.perf_counter() - self._raster_t0
    self._raster_t0 = None

  def _freeze_write_timer(self):
    if self._write_t0 is None:
      return
    self._write_elapsed = time.perf_counter() - self._write_t0
    self._write_t0 = None

  def _reset_current_file(self):
    self.file_features = 0
    self.file_features_done = 0
    self.file_pixels = 0
    self.feature_elapsed = 0.0
    self.count_seconds = 0.0
    self.count_window_px = 0
    self._feature_t0 = None
    self._raster_geoms_total = 0
    self._raster_geoms_done = 0
    self._raster_elapsed = 0.0
    self._raster_t0 = None
    self._write_rows_total = 0
    self._write_rows_done = 0
    self._write_elapsed = 0.0
    self._write_t0 = None

  def _refresh(self, force=False):
    snap = self.snapshot()
    now = time.perf_counter()
    if self._bar is not None and (force or now - self._last_bar >= _BAR_INTERVAL):
      self._bar.n = snap['percent']
      self._bar.set_description_str(PHASE_LABEL.get(snap['phase'], snap['phase']), refresh=False)
      self._bar.set_postfix_str(
        '{file} {done}/{total}'.format(
          file=snap['file'],
          done=snap['features_done'],
          total=snap['features_total'],
        ),
        refresh=False,
      )
      self._bar.refresh()
      self._last_bar = now
    if force or now - self._last_emit >= _EMIT_INTERVAL:
      tqdm.write(PROGRESS_PREFIX + json.dumps(snap, separators=(',', ':')), file=sys.stdout)
      sys.stdout.flush()
      self._last_emit = now
