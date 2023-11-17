# importing google_images_download module
from google_images_download import google_images_download 
 
# creating object
response = google_images_download.googleimagesdownload() 
 
men_queries_indian_attire = [
    "Nehru jacket designs for men",
    "Sherwani styles for weddings",
    "Kurta pajama for festive occasions",
    "Dhoti kurta traditional wear",
    "Pathani suit fashion",
    "Bandhgala suits for men",
    "Jodhpuri suits latest trends",
    "Men's ethnic wear for ceremonies",
    "Achkan sherwani for grooms",
    "Men's silk kurta for special events",
    "Safa turban styles",
    "Men's kurta designs for parties",
    "Churidar kurta fashion",
    "Traditional Indian wedding attire for men",
    "Men's Indo-Western fusion outfits",
    "Men's wedding sherwani collection",
    "Men's silk and brocade outfits",
    "Men's embroidered kurta for celebrations",
    "Men's ethnic jackets for occasions",
    "Men's traditional Indian clothing brands",
    "Men's traditional footwear for ethnic wear",
    "Men's festive wardrobe essentials",
    "Men's wedding accessories for traditional attire",
    "Stylish men's ethnic vests",
    "Men's Indian formal wear styles",
    "Indian men's fashion for business meetings",
    "Men's kurta with waistcoat for formal events",
    "Indian men's office wear fashion trends",
    "Men's ethnic blazers for functions",
    "Formal Indian menswear stores online"
]

women_queries_indian_attire = [
    "Saree draping styles for women",
    "Anarkali suit designs for festive occasions",
    "Lehenga choli for weddings",
    "Salwar kameez fashion trends",
    "Designer blouse patterns for sarees",
    "Patiala suits latest designs",
    "Bollywood style ethnic wear for women",
    "Women's ethnic gowns for special events",
    "Churidar suits for ceremonies",
    "Ethnic fusion wear for women",
    "Women's traditional Indian clothing brands",
    "Women's festive wear wardrobe essentials",
    "Embroidered sarees for celebrations",
    "Women's wedding lehenga collection",
    "Banarasi silk sarees for weddings",
    "Women's traditional Indian jewelry",
    "Women's dupatta styles for ethnic wear",
    "Women's ethnic footwear for occasions",
    "Women's traditional Indian makeup styles",
    "Women's Indian formal wear for business",
    "Women's office wear fashion in India",
    "Stylish women's ethnic jackets",
    "Women's Indian fashion accessories",
    "Women's festive saree collection",
    "Women's designer ethnic wear online",
    "Indian women's fashion for work meetings",
    "Women's designer salwar suits",
    "Women's silk and brocade outfits",
    "Indian women's clothing for professional settings",
    "Women's traditional Indian wear stores online"
]

 
def downloadimages(query, img_type):
    # keywords is the search query
    # format is the image file format
    # limit is the number of images to be downloaded
    # print urls is to print the image file url
    # size is the image size which can
    # be specified manually ("large, medium, icon")
    # aspect ratio denotes the height width ratio
    # of images to download. ("tall, square, wide, panoramic")
    arguments = {"keywords": query,
                 "format": "png",
                 "limit":1,
                 "print_urls":True,
                 "size": "medium",
                 "no_directory": True}
    try:
        response.download(arguments)
     
    # Handling File NotFound Error    
    except FileNotFoundError: 
        arguments = {"keywords": query,
                     "format": "png",
                     "limit":1,
                     "print_urls":True, 
                     "size": "medium",
                    #  "image_directory": "body-images/"+img_type,
                     "no_directory": True
                     }
                      
        # Providing arguments for the searched query
        try:
            response.download(arguments) 
        except Exception as e:
            print("Error:", e)

# Driver Code
for query in men_queries_indian_attire:
    downloadimages(query, "male") 
    print() 

# for query in women_queries_indian_attire:
#     downloadimages(query,"female") 
#     print() 