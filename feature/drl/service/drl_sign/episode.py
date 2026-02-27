from config import MODEL

class EpisodeService:
    def __init__(self):
        self.max_episode = MODEL.EPISODE_MEMORY
        self.episodes = []

    def add_episode(self, episode: dict):
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episode:
            self.episodes.pop(0)

    def run(self):
        return self.episodes
