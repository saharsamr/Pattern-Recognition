class DiscriminationOfFeatures(object):
    def __init__(self, feature_index, discrimination):
        self.feature_index = feature_index
        self.discrimination = discrimination

    def __ge__(self, other):
        return self.discrimination >= other.discrimination

    def __le__(self, other):
        return self.discrimination <= other.discrimination

    def __ne__(self, other):
        return self.discrimination != other.discrimination

    def __eq__(self, other):
        return self.discrimination == other.discrimination

    def __lt__(self, other):
        return self.discrimination < other.discrimination

    def __gt__(self, other):
        return self.discrimination > other.discrimination
