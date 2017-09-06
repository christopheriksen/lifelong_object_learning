import sys, string, math, time, socket
import flickrapi
from datetime import datetime
import xml.etree.ElementTree as ET
import urllib
from sets import Set

def main():
	api_key = "825ae511b1803f88cf3b7bf214ce00d8"
	api_secret = "11f47dbb08f23075"

	# query_strings = ["chair", "backpack", "officechair", "mug", "computerkeyboard"]
	query = "computerkeyboard"
	local_path = "/home/morgul/research_ws/src/lifelong_object_learning/data/scraped/"
	out_file = open(local_path + query + '.txt','w')
	num_iters = 1

	id_set = Set()

	flickr = flickrapi.FlickrAPI(api_key, api_secret)


	total_images_queried = 0;

	# number of seconds to skip per query  
	#timeskip = 62899200 #two years
	#timeskip = 604800  #one week
	timeskip = 172800  #two days
	#timeskip = 86400 #one day
	# timeskip = 3600 #one hour
	# timeskip = 2257 #for resuming previous query

	mintime = 1121832000 #from im2gps   7/20/2005
	#mintime = 1167407788 # resume crash england
	#mintime = 1177828976 #resume crash japan
	#mintime = 1187753798 #resume crash greece
	# mintime = 1171416400 #resume crash WashingtonDC
	maxtime = mintime+timeskip
	# endtime =  1192165200  #10/12/2007, at the end of im2gps queries
	endtime =  1499800647   #7/11/2017

	#this is the desired number of photos in each block
	# desired_photos = 250
	desired_photos = 50

	total_desired_photos = 1000



	num_acquired_phots = 0
	current_image_num = 0
	while ((maxtime < endtime) or (num_acquired_phots >= total_desired_photos)):

	    #new approach - adjust maxtime until we get the desired number of images 
	    #within a block. We'll need to keep upper bounds and lower
	    #lower bound is well defined (mintime), but upper bound is not. We can't 
	    #search all the way from endtime.

	    lower_bound = mintime + 900 #lower bound OF the upper time limit. must be at least 15 minutes or zero results
	    upper_bound = mintime + timeskip * 20 #upper bound of the upper time limit
	    maxtime     = .95 * lower_bound + .05 * upper_bound

	    print '\nBinary search on time range upper bound' 
	    print 'Lower bound is ' + str(datetime.fromtimestamp(lower_bound))
	    print 'Upper bound is ' + str(datetime.fromtimestamp(upper_bound))

	    keep_going = 6 #search stops after a fixed number of iterations
	    while( keep_going > 0 and maxtime < endtime):
	    
	        try:
	            rsp = flickr.photos.search(api_key=api_key,
	        			                        privacy_filter="1",
		                                        content_type="1",
		                                        per_page=str(desired_photos), 
		                                        page="1",
		                                        # has_geo = "1", #bbox="-180, -90, 180, 90",
		                                        text=query,
		                                        # accuracy="6", #6 is region level.  most things seem 10 or better.
		                                        min_upload_date=str(mintime),
		                                        max_upload_date=str(maxtime))
	            								##min_taken_date=str(datetime.fromtimestamp(mintime)),
	                                    		##max_taken_date=str(datetime.fromtimestamp(maxtime)))

	            #we want to catch these failures somehow and keep going.
	            time.sleep(1)
	            # fapi.testFailure(rsp)
	            photos = rsp[0]
	            total_images = photos.get('total')
	            null_test = int(total_images); #want to make sure this won't crash later on for some reason
	            null_test = float(total_images);
	    
	            print '\nnumimgs: ' + total_images
	            print 'mintime: ' + str(mintime) + ' maxtime: ' + str(maxtime) + ' timeskip:  ' + str(maxtime - mintime)
	        
	            if( int(total_images) > desired_photos ):
	                print 'too many photos in block, reducing maxtime'
	                upper_bound = maxtime
	                maxtime = (lower_bound + maxtime) / 2 #midpoint between current value and lower bound.
	            
	            if( int(total_images) < desired_photos):
	                print 'too few photos in block, increasing maxtime'
	                lower_bound = maxtime
	                maxtime = (upper_bound + maxtime) / 2
	            
	            print 'Lower bound is ' + str(datetime.fromtimestamp(lower_bound))
	            print 'Upper bound is ' + str(datetime.fromtimestamp(upper_bound))
	        
	            if( int(total_images) > 0): #only if we're not in a degenerate case
	                keep_going = keep_going - 1
	            else:
	                upper_bound = upper_bound + timeskip;    
	        
	        except KeyboardInterrupt:
	            print('Keyboard exception while querying for images, exiting\n')
	            raise
	        except:
	            print sys.exc_info()[0]
	            #print type(inst)     # the exception instance
	            #print inst.args      # arguments stored in .args
	            #print inst           # __str__ allows args to printed directly
	            print ('Exception encountered while querying for images\n')

	    #end of while binary search    
	    print 'finished binary search'

	    s = '\nmintime: ' + str(mintime) + ' maxtime: ' + str(maxtime)
	    print s
	    out_file.write(s + '\n') 

	    photos = rsp[0]
	    if photos != None:
	            
	        s = 'numimgs: ' + total_images
	        print s
	        out_file.write(s + '\n')
	        
	        num = int(photos.get('pages'))
	        s =  'total pages: ' + str(num)
	        print s
	        out_file.write(s + '\n')
	        
	        #only visit 16 pages max, to try and avoid the dreaded duplicate bug
	        #16 pages = 4000 images, should be duplicate safe.  Most interesting pictures will be taken.
	        
	        num_visit_pages = min(16,num)
	        
	        s = 'visiting only ' + str(num_visit_pages) + ' pages ( up to ' + str(num_visit_pages * desired_photos) + ' images)'
	        print s
	        out_file.write(s + '\n')
	        
	        total_images_queried = total_images_queried + min((num_visit_pages * desired_photos), int(total_images))

	        #print 'stopping before page ' + str(int(math.ceil(num/3) + 1)) + '\n'
	    
	        pagenum = 1;
	        while( pagenum <= num_visit_pages ):
	        #for pagenum in range(1, num_visit_pages + 1):  #page one is searched twice
	            print '  page number ' + str(pagenum)
	            try:
	                rsp = flickr.photos.search(api_key=api_key,

	                		                        privacy_filter="1",
			                                        content_type="1",
			                                        per_page=str(desired_photos), 
			                                        page=str(pagenum),
			                                        sort="interestingness-desc",
			                                        # has_geo = "1", #bbox="-180, -90, 180, 90",
			                                        text=query,
			                                        # accuracy="6", #6 is region level.  most things seem 10 or better.
			                                        extras = "tags, original_format, license, geo, date_taken, date_upload, o_dims, views",
			                                        min_upload_date=str(mintime),
			                                        max_upload_date=str(maxtime))

	                    
	                time.sleep(1)
	                # fapi.testFailure(rsp)
	            except KeyboardInterrupt:
	                print('Keyboard exception while querying for images, exiting\n')
	                raise
	            except:
	                print sys.exc_info()[0]
	                #print type(inst)     # the exception instance
	                #print inst.args      # arguments stored in .args
	                #print inst           # __str__ allows args to printed directly
	                print ('Exception encountered while querying for images\n')
	            else:

	                # and print them
	                photos = rsp[0]
	                if photos != None:
	    			    for child in photos:
					    	# print child.tag, child.attrib
					    	farm = str(child.get('farm'))
					    	server = str(child.get('server'))
					    	id_val = str(child.get('id'))
					    	secret = str(child.get('secret'))

					    	if not(id_val in id_set):

					    		url = "https://farm" + farm + ".staticflickr.com/" + server + "/" + id_val + "_" + secret + ".jpg"
			    				outfile = local_path + query + "/" + str(current_image_num) +".jpg"
			    				urllib.urlretrieve(url, outfile)

			    				id_set.add(id_val)
			    				current_image_num += 1
			    				num_acquired_phots += 1

	                pagenum = pagenum + 1;  #this is in the else exception block.  It won't increment for a failure.


	                if num_acquired_phots >= total_desired_photos:
	                	return

	        #this block is indented such that it will only run if there are no exceptions
	        #in the original query.  That means if there are exceptions, mintime won't be incremented
	        #and it will try again
	        timeskip = maxtime - mintime #used for initializing next binary search
	        mintime  = maxtime


if __name__ == "__main__":
    main()

