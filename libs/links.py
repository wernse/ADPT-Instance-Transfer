import requests
import re
import tldextract
short_links = [
    "bit.do", "t.co", "lnkd.in", "db.tt", "qr.ae", "adf.ly", "goo.gl",
    "bitly.com", "cur.lv", "tinyurl.com", "ow.ly", "bit.ly", "ity.im", "q.gs",
    "is.gd", "po.st", "bc.vc", "twitthis.com", "u.to", "j.mp", "buzurl.com",
    "cutt.us", "u.bb", "yourls.org", "x.co", "prettylinkpro.com", "scrnch.me",
    "filoops.info", "vzturl.com", "qr.net", "1url.com", "tweez.me", "v.gd",
    "tr.im", "link.zip.net", "tinyarrows.com"
]

# https://bit.ly/2LU3Wz4


def check_link(link, redirects, redirect_chain=""):
    # retweet case
    if link.startswith("https://twitter.com"):
        return (link, None, None)

    if redirects == 0:
        redirect_chain = link

    # Set max limit on the redirects to prevent infinite loop
    if redirects > 20:
        return (link, redirects, redirect_chain)

    # connect domain and sub doamin
    extract = tldextract.extract(link)
    domain_url = '{}.{}'.format(extract.domain, extract.suffix)
    # if domain_url is in short_links list then repeat
    if domain_url in short_links:
        r = requests.get(link, allow_redirects=False)

        # outlier case
        if r.status_code != 301:
            return (link, redirects, redirect_chain)

        # Redirection case if shortened link
        if r.status_code == 301:
            redirect_chain = "{}, {}".format(redirect_chain,
                                             r.headers['location'])
            redirects = redirects + 1
            return check_link(r.headers['location'], redirects, redirect_chain)
    else:
        return (link, redirects, redirect_chain)


def get_links_text(text):
    return re.findall('https?://t.co/[a-zA-Z0-9]{10}', text)


def get_link_domain(link):
    extract = tldextract.extract(link)
    return extract.domain


def get_without_prefix(link):
    import re
    url = re.compile(r"https?://(www\.)?")
    formatted_url = url.sub('',
                            link).strip().strip('/').replace('.', '').replace(
                                '/', '').replace(':', '')
    return formatted_url