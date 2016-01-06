---
layout: post
title: "ElasticSearch Query DSL"
date: "2016-01-06 14:17"
---

### Elasticsearach QueryDSL

Elasticsearch provides a full Java query dsl in a similar manner to the REST Query DSL. The factory for query builders is QueryBuilders. Once your query is ready, you can use the Search API.

To use QueryBuilders just import them in your class


~~~ java
import static org.elasticsearch.index.query.QueryBuilders.*;
~~~
Note that you can easily print (aka debug) JSON generated queries using toString() method on QueryBuilder object.

The QueryBuilder can then be used with any API that accepts a query, such as count and search.

# Full Text queries
---

The queries in this group are:

match query
:The standard query for performing full text queries, including fuzzy matching and phrase or proximity queries.

multi_match query
:The multi-field version of the match query.

common_terms query
:A more specialized query which gives more preference to uncommon words.

query_string query
:Supports the compact Lucene query string syntax, allowing you to specify AND|OR|NOT conditions and multi-field search within a single query string. For expert users only.

simple_query_string
:A simpler, more robust version of the query_string syntax suitable for exposing directly to users.

#Lucene Architecture

![Lucene Architecture]({{ site.url }}/images/2016/01/lucene_pic.jpeg)
