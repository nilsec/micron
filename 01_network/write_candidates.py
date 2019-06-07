from funlib import math
import gunpowder as gp
import numpy as np
import pymongo
import logging

logger = logging.getLogger(__name__)


class WriteCandidates(gp.BatchFilter):

    def __init__(
            self,
            maxima,
            db_host,
            db_name):

        self.maxima = maxima
        self.db_host = db_host
        self.db_name = db_name
        self.client = None

    def process(self, batch, request):

        if self.client is None:
            self.client = pymongo.MongoClient(host=self.db_host)
            self.db = self.client[self.db_name]
            create_indices = 'nodes' not in self.db.list_collection_names()
            self.candidates = self.db['nodes']
            if create_indices:
                self.candidates.create_index(
                    [
                        (l, pymongo.ASCENDING)
                        for l in ['z', 'y', 'x']
                    ],
                    name='position')
                self.candidates.create_index(
                    [
                        ('id', pymongo.ASCENDING)
                    ],
                    name='id',
                    unique=True)

        roi = batch[self.maxima].spec.roi
        voxel_size = batch[self.maxima].spec.voxel_size

        maxima = batch[self.maxima].data

        candidates = []
        for index in np.argwhere(maxima>0.5):

            index = gp.Coordinate(index)
            position = roi.get_begin() + voxel_size*index

            candidate_id = int(math.cantor_number(
                roi.get_begin()/voxel_size + index))

            candidates.append({
                'id': candidate_id,
                'z': position[0],
                'y': position[1],
                'x': position[2],
	        'degree': 0,
                'selected': False,
                'solved': False
            })

            logger.debug(
                "ID=%d" % candidate_id)

        if len(candidates) > 0:
            self.candidates.insert_many(cells)
