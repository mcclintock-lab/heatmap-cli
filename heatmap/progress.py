import sys
from tqdm import tqdm


def create_progress(total, desc, unit='it', initial=0):
  return tqdm(
    total=max(int(total), 0),
    desc=desc,
    unit=unit,
    initial=initial,
    file=sys.stderr,
    ncols=120,
    leave=False,
    smoothing=0.05,
    mininterval=0.1,
  )


def progress_write(_progress, msg):
  print(msg)


class FeatureProgress:
  """A feature-count bar that can close between files so later prints stay clean."""

  def __init__(self, total):
    self.total = max(int(total), 0)
    self.n = 0
    self._bar = None

  def start(self, postfix=''):
    if self._bar is not None:
      return
    self._bar = create_progress(
      self.total,
      'Processing Features',
      unit='feat',
      initial=self.n,
    )
    if postfix:
      self._bar.set_postfix_str(postfix, refresh=True)

  def update(self, n=1):
    self.n += n
    if self._bar is not None:
      self._bar.update(n)

  def finish(self):
    if self._bar is None:
      return
    self._bar.close()
    self._bar = None
