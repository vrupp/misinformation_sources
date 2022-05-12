# CrowdTangle
* CrowdTangle provides access over [UI](https://apps.crowdtangle.com/garcialab) and [API](https://github.com/CrowdTangle/API)

## UI
### Usage
* https://apps.crowdtangle.com/garcialab
* You can create Lists (of pages or groups, not both)
* You can create Searches filtering for terms, Lists, Pages, min. Page follower, language
  * Exclusion of terms, Lists, Pages also possible
* Historical Data
  * Settings > Historical Data
  * Searches for posts in CrowdTangle database, not Facebook API
  * Filter option (step 2) has very limited features, e.g. filter by term and post type
  * Scope (step 1) let you select a saved Search (and List, Pages, Groups) and thus gives you much more filter options
  * Has a limit around 300000 (trial and error). Can be a couple less (for example 299970).
* If the search query is flawed, CrowdTangle seems to just deliver a random (not reproducible) number of tweets. There is NO warning if the query fails.
* Chaining Boolean queries like (a OR b) AND (c OR d) seems to work
  * Comma-separating works as OR
* CrowdTangle seems to be not case sensitive, i.e. searching for "soros" also yields posts containing "Soros".
* Query string is restricted to 1024 clauses (trial and error) (server response: "The query has too many clauses. Please simplify and try again.")
* Post limit for non-historical data is 10.000.

### Response Type
* CSV
* For a request you get an email with a link to the CSV file
* Header
  * **Page Info**: Page Name, User Name, Facebook Id, Page Category, Page Admin Top Country, Page Description, Page Created, Likes at Posting, Followers at Posting
  * **Post Metadata**: Post Created, Post Created Date, Post Created Time, Type, Total Interactions, Likes, Comments, Shares, Love, Wow, Haha, Sad, Angry, Care, Video Share Status, Is Video Owner?, Post Views, Total Views, Total Views For All Crossposts, Video Length, URL
  * **Post Content**: Message, Link, Final Link, Image Text, Link Text, Description
  * **Sponsor Info**: Sponsor Id, Sponsor Name, Sponsor Category, Overperforming Score (weighted  —  Likes 1x Shares 1x Comments 1x Love 1x Wow 1x Haha 1x Sad 1x Angry 1x Care 1x )

## API
* https://github.com/CrowdTangle/API
  * [API Cheat Sheet](https://help.crowdtangle.com/en/articles/3443476-api-cheat-sheet)
* Get API Key via Settings > API Access
* Base URL: https://api.crowdtangle.com
* `/posts` endpoint can search for search terms in Facebook posts. With parameter `listIds` you can specify IDs of Lists (created in the UI) or a saved search (also created in the UI). Without that parameter only those Pages from your Lists are searched
* `/posts/search` is like Historic Data in the UI, but needs prior approval by CrowdTangle
* `/links` searches for posts matching a link. This seems more powerful, since redirects are included. It also includes posts from Facebook, Instagram, and Reddit. But `link` parameter only allows a single link
  * It seems that also ImageText (probably generated) is searched
    * ![](ct-image-text.png)
* Post limit of `/posts` is actually 2 million, but I received warning after ~70k posts: "The search exceeded the maximum number of hits in the timeframe given. This maximum number of possible results is 2000000. The actual start date used in this query can be accessed via the actualStartDate property in the response. To access all of the available results, please make multiple requests with smaller timeframes or continue making requests replacing the endDate of the next request with the actualStartDate returned here."
* Pro
  * More search options than in UI, e.g. start/end date
  * Detailed documentation of usage and response format
  * Seems to have no total limit (only limits per request, pagination gets next batch)
* Con
  * Pagination looks tedious: 
    * `count` parameter determines number of posts in response (restricted to `1-100`)
    * With `offset` parameter one can request the next batch
    * e.g. `count=100&offset=0` gives you post 1-100, `count=100&offset=100` gives you post 101-200, etc.

### Response Type
* JSON or XML
* Format distinguishes by usage