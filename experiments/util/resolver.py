import os
import random
from omegaconf import OmegaConf

def random_name():
  path = os.path.join(os.path.dirname(__file__), 'words', 'adjectives.txt')
  with open(path) as fp:
    adjectives = fp.read().splitlines()
  path = os.path.join(os.path.dirname(__file__), 'words', 'nouns.txt')
  with open(path) as fp:
    nouns = fp.read().splitlines()
  return '-'.join(random.choices(adjectives) + random.choices(nouns))

def init_resolver():
  name = random_name()
  OmegaConf.register_new_resolver('eval', eval)
  OmegaConf.register_new_resolver('random_name', lambda: name)
