from pocket import Pocket

ckey = '47167-9324a229a155827bc214aa2a'
redir = 'http://google.com'

request_token = Pocket.get_request_token(consumer_key=ckey, redirect_uri=redir)

# URL to redirect user to, to authorize your app
auth_url = Pocket.get_auth_url(code=request_token, redirect_uri=redir)
print(auth_url)
input()

user_credentials = Pocket.get_credentials(consumer_key=ckey, code=request_token)

access_token = user_credentials['access_token']

pocketi = Pocket(ckey, access_token)


for link in links[500:]:
    pocketi.add(link)

import numpy as np
import matplotlib.pyplot as plt

xaxis = range(1,50)
plt.plot(xaxis, [np.sqrt(6*sum([1./n**2 for n in range(1,x)])) for x in xaxis])
plt.axhline(np.pi, color='r')
plt.show()z
