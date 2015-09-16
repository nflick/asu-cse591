#!/usr/bin/env python

import json
import os
import pwd
import signal
import sys
import traceback
from time import sleep
from collections import deque
from random import sample
from datetime import datetime

import requests
import sqlalchemy as sql
import sqlalchemy.orm as orm
import sqlalchemy.dialects.postgresql as psql
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = sql.Column(sql.BigInteger, primary_key=True)
    next_max_id = sql.Column(sql.String)
    fully_scraped = sql.Column(sql.Boolean, default=False)

    def __repr__(self):
        return "<User(id={0}, next_url='{1}')>".format(self.id, self.next_url)

class Image(Base):
    __tablename__ = 'images'

    id = sql.Column(sql.BigInteger, primary_key=True)
    date = sql.Column(sql.DateTime)
    caption = sql.Column(sql.String)
    tags = sql.Column(psql.ARRAY(sql.String))
    lat = sql.Column(sql.Float)
    lng = sql.Column(sql.Float)
    user_id = sql.Column(sql.BigInteger, sql.ForeignKey('users.id'))
    user = orm.relationship('User', backref=orm.backref('images', order_by=id))

    def __repr__(self):
        return "<Image(id={0}, date='{1}', tags='{2}', lat={3}, lng={4})>".format(
            self.id, self.date, self.tags, self.lat, self.lng)

def attempt(func):
    try:
        func()
    except:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.write('\n')

class Crawler:
    def __init__(self, session, verbose, max_branching=5, queue_size=5000):
        self.session = session
        self.verbose = verbose
        self.max_branching = max_branching
        # Queue contains only user ID's
        self.queue = deque(maxlen=queue_size)
        self.stop = False

    def run(self):
        '''Crawls the target site by performing a hybrid breadth-first/depth-first
        traversal of the users, collecting image metadata from each user.
        Assumes the crawler has already been seeded.
        '''
        self.stop = False
        while len(self.queue) > 0 and not self.stop:
            user_id = self.queue.popleft()
            user = self.session.query(User).get(user_id)
            if user is None:
                user = User(id=user_id)
                self.session.add(user)
            attempt(lambda: self.branch(user))
            attempt(lambda: self.scrape(user))
            self.session.commit()

    def branch(self, user):
        '''Adds to the search queue by selecting from the successors
        of this user.
        '''
        successors = self.successors(user)
        limit = max(self.max_branching, self.queue.maxlen - len(self.queue))
        if len(successors) > limit:
            self.queue.extend(sample(successors, limit))
        else:
            self.queue.extend(successors)

class InstagramCrawler(Crawler):
    api_base = 'https://api.instagram.com/v1'
    max_images = 100

    def __init__(self, session, client_id, verbose, max_branching=5, queue_size=5000):
        super().__init__(session, verbose, max_branching, queue_size)
        self.client_id = client_id

    def api_request(self, endpoint, params={}):
        '''Sends an Instagram API request, throttling as necessary to avoid
        sending more requests than alloted, and returns the parsed JSON.
        Prints the URL and response code if set to verbose.
        '''
        url = self.api_base + endpoint if endpoint.startswith('/') else '{0}/{1}'.format(self.api_base, endpoint)
        params['client_id'] = self.client_id
        resp = requests.get(url, params)
        content = json.loads(resp.content.decode())
        if self.verbose:
            print(resp.url[len(self.api_base):])
            print(resp.status_code)
        # Avoid exceeding the Instagram API limits and getting access turned off.
        if resp.status_code == 429 or ('x-ratelimit-remaining' in resp.headers and int(resp.headers['x-ratelimit-remaining']) < 100):
            if self.verbose:
                print("Pausing to throttle API calls...")
            time.sleep(60)
        return content

    def successors(self, user):
        '''Returns a list of users connected to the specified user. Currently
        this just returns the user's followers.
        '''
        content = self.api_request('/users/{0}/followed-by'.format(user.id))
        if content['meta']['code'] == 200:
            return [int(u['id']) for u in content['data']]
        return []

    def scrape(self, user):
        '''Scrapes the recent images of the user. Scrapes only a certain number
        of images at a time; if scrape is called again in the future on this
        user the next set of images will be scraped.
        '''
        if user.fully_scraped:
            return
        params = {'count': self.max_images}
        if user.next_max_id is not None:
            params['max_id'] = user.next_max_id

        content = self.api_request('/users/{0}/media/recent'.format(user.id), params)
        if content['meta']['code'] == 200:
            if 'next_max_id' in content['pagination']:
                user.next_max_id = content['pagination']['next_max_id']
            else:
                user.next_max_id = None
                user.fully_scraped = True
            for media in content['data']:
                self.store(user, media)


    def store(self, user, media):
        '''Determines if the given media contains location information. If so,
        the date, caption, tags, and location are recorded into the database.
        '''
        if media['location'] is not None and 'latitude' in media['location'] and 'longitude' in media['location']:
            id_ = int(media['id'].split('_')[0])
            image = self.session.query(Image).get(id_)
            if image is None:
                if media['caption'] is not None and 'text' in media['caption']:
                    caption = media['caption']['text']
                else:
                    caption = None
                image = Image(id=id_,
                    date=datetime.fromtimestamp(int(media['created_time'])),
                    caption=caption,
                    tags=media['tags'],
                    lat=float(media['location']['latitude']),
                    lng=float(media['location']['longitude']),
                    user=user)
                self.session.add(image) 

    def seed_by_popular(self):
        '''Uses the /media/popular API endpoint to initialize the search queue
        with a number of users.
        '''
        content = self.api_request('/media/popular')
        if content['meta']['code'] != 200:
            for media in content['data']:
                self.queue.append(int(media['user']['id']))
        else:
            raise Exception('Response code {0} received.'.format(content['meta']['code']))

    def seed_by_location(self, lat, lng):
        '''Uses the /media/search API endpoint to initialize the search queue
        with a number of users.
        '''
        params = {'lat': lat, 'lng': lng}
        content = self.api_request('/media/search', params)
        if content['meta']['code'] == 200:
            for media in content['data']:
                self.queue.append(int(media['user']['id']))
        else:
            raise Exception('Response code {0} received.'.format(content['meta']['code']))

    def seed_from_database(self):
        '''Seeds the queue with the users associated with the most recent images
        (by date/time taken) in the database.
        '''
        images = self.session.query(Image).options(orm.joinedload(Image.user)).\
            order_by(sql.desc(Image.date)).limit(self.queue.maxlen)
        self.queue.extend(set(im.user.id for im in images))

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Verbose output.')
    parser.add_argument('-c', '--create-tables', action='store_true',
        help='Create database tables.')
    parser.add_argument('-p', '--seed-popular', action='store_true',
        help='Seed crawler with popular users.')
    parser.add_argument('-l', '--seed-location', metavar='LAT,LONG',
        help='Seed crawler with recent users at the given location.')
    parser.add_argument('-s', '--seed-database', action='store_true',
        help='Seed the crawler with users from the database.')
    parser.add_argument('-d', '--dbhost', default='localhost',
        help='Database host.')
    parser.add_argument('-u', '--dbuser', help='Database user name.')
    parser.add_argument('database', help='Specify database name.')
    args = parser.parse_args()

    # Additional validation
    #if args.seed_location is None and not args.seed_popular:
    #    parser.error('One seed method must be provided.')

    if args.seed_location is not None:
        try:
            args.seed_location_split = [float(c) for c in args.seed_location.split(',')]
            assert(len(args.seed_location_split) == 2)
        except:
            parser.error('Invalid format for seed location: {0}'.format(args.seed_location))
    else:
        args.seed_location_split = None
    
    if args.dbuser is None:
        args.dbuser = pwd.getpwuid(os.getuid()).pw_name

    if 'DBPASS' in os.environ:
        args.dbpass = os.environ['DBPASS']
    else:
        args.dbpass = None

    if 'INSTAGRAM_CLIENT_ID' not in os.environ:
        parser.error('INSTAGRAM_CLIENT_ID must be provided as an environmental variable.')
    args.client_id = os.environ['INSTAGRAM_CLIENT_ID']

    return args

def stop_crawler(crawler):
    print('Stopping crawler...')
    crawler.stop = True

def main():
    args = get_args()
    if args.dbpass is None:
        connstring = 'postgresql://{0}@{1}/{2}'.format(args.dbuser, args.dbhost, args.database)
    else:
        connstring = 'postgresql://{0}:{1}@{2}/{3}'.format(args.dbuser, args.dbpass, args.dbhost, args.database)
    engine = sql.create_engine(connstring)

    if args.create_tables:
        print('Creating tables...')
        Base.metadata.create_all(engine)
        return

    Session = orm.sessionmaker(bind=engine)
    session = Session()
    crawler = InstagramCrawler(session, args.client_id, args.verbose)

    if args.seed_popular:
        print('Seeding popular content...')
        crawler.seed_by_popular()
    if args.seed_location is not None:
        print('Seeding by location...')
        crawler.seed_by_location(args.seed_location_split[0], args.seed_location_split[1])
    if args.seed_database:
        print('Seeding from database...')
        crawler.seed_from_database()

    # Setup handler so that the interrupt signal can be caught and
    # the crawler can exit cleanly.
    signal.signal(signal.SIGINT, lambda s, f: stop_crawler(crawler))
    crawler.run()
    session.close()

if __name__ == '__main__':
    main()
