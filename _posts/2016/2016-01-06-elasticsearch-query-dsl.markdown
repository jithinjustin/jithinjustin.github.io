---
layout: post
title: "ElasticSearch Query DSL"
date: "2016-01-06 14:17"
---


Elasticsearch provides a full Java query dsl in a similar manner to the REST Query DSL. The factory for query builders is QueryBuilders. Once your query is ready, you can use the Search API. Following are taken from [official documenation][73babb6a] for Elasticsearch version 2.1.

  [73babb6a]: https://www.elastic.co/guide/en/elasticsearch/client/java-api/current/java-query-dsl.html "es documentation 2.1"

To use QueryBuilders just import them in your class


{% highlight java %}
import static org.elasticsearch.index.query.QueryBuilders.*;
{% endhighlight %}
Note that you can easily print (aka debug) JSON generated queries using toString() method on QueryBuilder object.

The QueryBuilder can then be used with any API that accepts a query, such as count and search.

# Full Text queries
---

The queries in this group are:

* match query - The standard query for performing full text queries, including fuzzy matching and phrase or proximity queries.

{% highlight java %}
QueryBuilder qb = matchQuery(
    "name",     //field             
    "kimchy elasticsearch"   //text
);
{% endhighlight %}



* multi_match query - The multi-field version of the match query.

{% highlight java %}
QueryBuilder qb = multiMatchQuery(
    "kimchy elasticsearch", //text
    "user", "message"       //fields
);
{% endhighlight %}

* common_terms query - A more specialized query which gives more preference to uncommon words.

{% highlight java %}
QueryBuilder qb = commonTermsQuery("name",    //field
                                   "kimchy"); //value
{% endhighlight %}

* query_string query - Supports the compact Lucene query string syntax, allowing you to specify AND or OR or NOT conditions and multi-field search within a single query string. For expert users only.

{% highlight java %}
QueryBuilder qb = queryStringQuery("+kimchy -elasticsearch"); //text
{% endhighlight %}

* simple_query_string : A simpler, more robust version of the query_string syntax suitable for exposing directly to users.

{% highlight java %}
QueryBuilder qb = simpleQueryStringQuery("+kimchy -elasticsearch"); //text
{% endhighlight %}
