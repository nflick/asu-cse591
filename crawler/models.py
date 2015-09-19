# ASU CSE 591
# Author: Nathan Flick

import sqlalchemy as sql
import sqlalchemy.orm as orm
import sqlalchemy.sql.expression as sqlexpr
import sqlalchemy.dialects.postgresql as psql
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = sql.Column(sql.BigInteger, primary_key=True)
    next_max_id = sql.Column(sql.String)
    fully_scraped = sql.Column(sql.Boolean, default=False)
    private = sql.Column(sql.Boolean, default=False)

    def __repr__(self):
        return "<User(id={0}, next_max_id='{1}' fully_scraped={2}, private={3})>".format(
        	self.id, self.next_max_id, self.fully_scraped, self.private)

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

def create_engine(connstring):
	return sql.create_engine(connstring)

def create_session(engine):
	return orm.sessionmaker(bind=engine)()

def create_tables(engine):
	Base.metadata.create_all(engine)

def random_users(session, num):
	# Postgres specific
    return session.query(User).order_by(sqlexpr.func.random()).limit(num).all()