import numpy as np
import numpy.linalg

class FontPropertiesMaker:
    def __init__(self, db):
        self.db = db

    def cosine_extractor(self, vec1, vec2):
        return np.dot(vec1, vec2) / (numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2))

    def fontsizemaker(self, id1, id2, pop, hot):
        relevance = self.cosine_extractor(self.db[['id']==id1]['embeddings'], self.db[['id']==id2]['embeddings'])
        return relevance*0.25+pop*0.5+hot*0.25
    
    def radiusmaker(self, id1, id2):
        relevance = self.cosine_extractor(self.db[['id']==id1]['embeddings'], self.db[['id']==id2]['embeddings'])
        return relevance