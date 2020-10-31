import pymongo

def get_daisy_collection_name(step_name):
    return step_name + "_daisy"


def check_function(db, block, step_name):
    daisy_coll = db[get_daisy_collection_name(step_name)]
    result = daisy_coll.find_one({'_id': block.block_id})
    if result is None:
        return False
    else:
        return True

def write_done(db, block, step_name):
    daisy_coll = db[get_daisy_collection_name(step_name)]
    daisy_coll.insert_one({'_id': block.block_id})

def attr_exists(db_name, db_host, collection, attr):
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    coll = db[collection]
    num_exists = coll.count_documents({attr : {"$exists": True}})
    if num_exists == 0:
        return False
    else:
        return True

def reset_step(step_name, db_name, db_host):
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    daisy_coll = db[get_daisy_collection_name(step_name)]
    daisy_coll.drop()

def reset_solve(db_name, db_host, edge_collection, node_collection, selected_attr, solved_attr):
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    edges = db[edge_collection].update_many({}, {"$set": {selected_attr: False, solved_attr: False}})
    nodes = db[node_collection].update_many({}, {"$set": {selected_attr: False, solved_attr: False}})
