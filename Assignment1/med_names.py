
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
"""
Importing library for parsing the HTML parser
"""

"""
This function makes a GET Request to the specified URL and returns the response
"""
def simple_get(url):
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be the desired HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    print(e)


"""

Scraper
Target URL - https://www.drugs.com

This function hits the target URL and fetches all medicines starting with the given characters 
as specified in the parameter input

Return type: dict {'medicine name':'target sub-url'}

"""

""" 
This function parses the index page of the website and 
returns the names of all medicines
"""
def get_med_names(letter):
    url = 'https://www.drugs.com '+ letter
    print("med list  " +letter)
    response = simple_get(url)
    if response is not None:
        html = BeautifulSoup(response, 'html.parser')
        medicines = {}
        html2 = html.find('ul', {'class': 'ddc-list-column-2'})
        if html2 is None:
            html2 = html.find('ul', {'class': 'ddc-list-unstyled'})
        if html2 is not None:
            for x in html2.find_all('li'):
                medicines[x.text] = 'https://www.drugs.com' + x.a['href']
        return medicines
    else:
        return {}

""" 
This function returns the set of first two letters of 
all medicines available in the website
"""
def get_all_initial_letters():
    letter s =[]
    for i in range(26):
        print('letter= ' +chr( i +97))
        root_res p =simple_get("https://www.drugs.com/alpha/ " +chr( i +97 ) +".html")
        if root_resp is None:
            continue
        else:
            html = BeautifulSoup(root_resp, 'html.parser')
            for x in html.find('ul', {'class': 'ddc-paging'}).find_all('li'):
                if x.a is not None:
                    letters.append(x.a['href'])
    print(letters)
    return letters

# MAIN
if __name__ == '__main__':
    """
    This file shall store the list of all the medicines and their URL where the medicine information is available f= open("all_medicines.txt","a")
    """
    for x in get_all_initial_letters():
        med_names = get_med_names(x)
        for med in med_names:
            try:
                f.write(med+"=" + med _ names[med]+"\n " )
            except:
                print('error in writing')
    f.close()

