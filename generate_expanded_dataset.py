import random

# Define topics and sample queries
topics = [
    "health", "technology", "food", "travel", "education", 
    "finance", "sports", "entertainment", "science", "history"
]

# Generate sample queries for each topic (5 per topic)
queries = {
    "health": [
        "what is diabetes",
        "how to lower blood pressure",
        "benefits of exercise",
        "covid symptoms",
        "treating common cold"
    ],
    "technology": [
        "python list comprehension",
        "machine learning basics",
        "how to build a website",
        "what is cloud computing",
        "smartphone comparison"
    ],
    "food": [
        "healthy breakfast ideas",
        "how to cook pasta",
        "vegan protein sources",
        "best coffee brewing methods",
        "easy dinner recipes"
    ],
    "travel": [
        "weather in london",
        "best european cities",
        "travel packing tips",
        "new york attractions",
        "budget travel destinations"
    ],
    "education": [
        "effective study techniques",
        "online learning platforms",
        "how to learn languages",
        "math tutoring resources",
        "college application tips"
    ],
    "finance": [
        "investing for beginners",
        "how to save money",
        "understanding credit scores",
        "retirement planning basics",
        "tax deduction strategies"
    ],
    "sports": [
        "basketball rules explained",
        "world cup history",
        "training for a marathon",
        "tennis techniques",
        "olympic games facts"
    ],
    "entertainment": [
        "popular movie genres",
        "benefits of meditation",
        "book recommendations",
        "music streaming services",
        "video game consoles"
    ],
    "science": [
        "how solar panels work",
        "basics of genetics",
        "climate change explanation",
        "space exploration history",
        "scientific method"
    ],
    "history": [
        "ancient egyptian civilization",
        "world war 2 overview",
        "renaissance period",
        "industrial revolution",
        "civil rights movement"
    ]
}

# Generate relevant (positive) answers for each query
relevant_answers = {
    "what is diabetes": "Diabetes is a chronic disease that occurs when the pancreas cannot produce enough insulin or when the body cannot effectively use the insulin it produces.",
    "how to lower blood pressure": "Lowering blood pressure can be achieved through regular exercise, reducing salt intake, maintaining a healthy weight, limiting alcohol, and possibly medication prescribed by a doctor.",
    "benefits of exercise": "Regular exercise strengthens the heart, improves lung function, reduces stress, helps with weight management, and lowers risk of chronic diseases like diabetes and heart disease.",
    "covid symptoms": "Common COVID-19 symptoms include fever, cough, fatigue, loss of taste or smell, sore throat, headache, and shortness of breath, though severity varies widely among individuals.",
    "treating common cold": "Common cold treatment includes rest, staying hydrated, over-the-counter pain relievers, decongestants, and allowing the viral infection to run its course typically within 7-10 days.",
    
    "python list comprehension": "In Python, a list comprehension is a concise way to create lists using a single line of code.",
    "machine learning basics": "Machine learning is a field of AI that enables computers to learn from data and make predictions without explicit programming, using algorithms that improve through experience.",
    "how to build a website": "Building a website involves choosing a domain name, selecting a web hosting service, designing the layout, creating content, and either coding it or using a content management system.",
    "what is cloud computing": "Cloud computing delivers computing services including servers, storage, databases, networking, software, and analytics over the internet ('the cloud') for faster innovation and flexible resources.",
    "smartphone comparison": "Smartphone comparisons evaluate devices based on processing power, camera quality, battery life, display resolution, storage capacity, price, and operating system features.",
    
    "healthy breakfast ideas": "Healthy breakfast options include oatmeal with fruit, Greek yogurt with granola, whole grain toast with avocado, smoothie bowls, and egg-based dishes rich in protein.",
    "how to cook pasta": "Cooking pasta involves boiling water with salt, adding pasta, stirring occasionally, cooking until al dente (firm to the bite), and draining before adding sauce.",
    "vegan protein sources": "Vegan protein sources include legumes (beans, lentils), tofu, tempeh, seitan, quinoa, nuts, seeds, and various plant-based protein powders made from pea, rice, or hemp.",
    "best coffee brewing methods": "Popular coffee brewing methods include pour-over, French press, espresso, AeroPress, cold brew, and drip coffee, each producing different flavor profiles and strengths.",
    "easy dinner recipes": "Easy dinner recipes include one-pot pasta dishes, sheet pan meals, stir-fries, loaded salads, grain bowls, and simple protein with vegetable combinations.",
    
    "weather in london": "London has a temperate maritime climate with mild summers and cool winters, and rainfall distributed fairly evenly throughout the year.",
    "best european cities": "Popular European cities for tourism include Paris, Rome, Barcelona, Amsterdam, Prague, Vienna, Budapest, and Berlin, each offering unique cultural experiences and historical sites.",
    "travel packing tips": "Effective packing strategies include making a list, rolling clothes, using packing cubes, packing versatile clothing, and limiting footwear to save space.",
    "new york attractions": "Top New York City attractions include the Statue of Liberty, Empire State Building, Central Park, Times Square, Metropolitan Museum of Art, and Broadway shows.",
    "budget travel destinations": "Affordable international destinations include Vietnam, Thailand, Mexico, Portugal, Turkey, and Eastern European countries where accommodations and daily expenses are lower.",
    
    "effective study techniques": "Effective studying includes spaced repetition, active recall, teaching concepts to others, breaking material into chunks, and studying in short, focused sessions with breaks.",
    "online learning platforms": "Popular online learning platforms include Coursera, Udemy, Khan Academy, edX, LinkedIn Learning, and Skillshare, offering courses across numerous subjects and skill levels.",
    "how to learn languages": "Language learning strategies include daily practice, immersion, conversation with native speakers, using apps like Duolingo, consuming media in the target language, and focused vocabulary building.",
    "math tutoring resources": "Math learning resources include Khan Academy, Brilliant.org, Desmos, YouTube tutorials, online tutoring services, and interactive problem-solving websites.",
    "college application tips": "Successful college applications require planning ahead, maintaining strong academics, pursuing meaningful extracurriculars, writing compelling essays, and securing strong recommendation letters.",
    
    "investing for beginners": "Beginning investors should focus on understanding investment types, setting clear goals, starting with index funds or ETFs, diversifying, and investing consistently over time.",
    "how to save money": "Effective money-saving strategies include creating a budget, reducing unnecessary expenses, automating savings, comparing prices before purchases, and setting specific financial goals.",
    "understanding credit scores": "Credit scores are numerical ratings based on payment history, credit utilization, length of credit history, new credit applications, and types of credit used.",
    "retirement planning basics": "Retirement planning involves estimating expenses, identifying income sources, establishing savings strategy, managing assets, creating an estate plan, and maximizing tax-advantaged accounts.",
    "tax deduction strategies": "Common tax deductions include mortgage interest, charitable contributions, medical expenses, retirement contributions, education expenses, and business-related deductions if applicable.",
    
    "basketball rules explained": "Basketball rules include scoring (2 or 3 points for field goals, 1 point for free throws), dribbling requirements, fouls, violations like traveling and double-dribbling, and time restrictions.",
    "world cup history": "The FIFA World Cup began in 1930, occurs every four years, features 32 national teams (expanding to 48 in 2026), and has been dominated historically by Brazil with five championships.",
    "training for a marathon": "Marathon training typically includes progressive long runs, speed work, cross-training, proper nutrition, adequate recovery, and a 16-20 week structured program building up to the race distance.",
    "tennis techniques": "Fundamental tennis techniques include the forehand, backhand, serve, volley, proper footwork, and developing consistency through practice of proper form and strategy.",
    "olympic games facts": "The modern Olympic Games began in 1896, feature summer and winter competitions held every four years, include over 200 nations, and celebrate athletic excellence across dozens of sports.",
    
    "popular movie genres": "Major film genres include drama, comedy, action, thriller, horror, science fiction, fantasy, romance, documentary, and animation, each with distinct storytelling conventions and visual styles.",
    "benefits of meditation": "Regular meditation practice can reduce stress, enhance concentration, and improve overall emotional well‑being.",
    "book recommendations": "Popular books span genres like fiction (novels, short stories), non-fiction (biographies, self-help, history), with recommendations often based on interests, reading level, and current bestsellers.",
    "music streaming services": "Music streaming platforms include Spotify, Apple Music, Amazon Music, YouTube Music, and Tidal, offering vast libraries with different pricing tiers and audio quality options.",
    "video game consoles": "Major gaming consoles include PlayStation, Xbox, and Nintendo Switch, each offering exclusive games, online services, and varying technical capabilities for different gaming experiences.",
    
    "how solar panels work": "Solar panels convert sunlight into electricity through photovoltaic cells containing silicon that release electrons when struck by photons, creating direct current that's converted to usable alternating current.",
    "basics of genetics": "Genetics studies how traits are passed from parents to offspring through DNA, genes, and chromosomes, with principles of inheritance following patterns first described by Gregor Mendel.",
    "climate change explanation": "Climate change refers to long-term shifts in temperatures and weather patterns driven primarily by human activities, especially fossil fuel burning that increases heat-trapping greenhouse gases in Earth's atmosphere.",
    "space exploration history": "Space exploration began with Sputnik in 1957, followed by human spaceflight starting with Yuri Gagarin in 1961, the Apollo moon landings, space stations, and modern exploration by both nations and private companies.",
    "scientific method": "The scientific method involves making observations, forming hypotheses, conducting experiments to test predictions, analyzing results, drawing conclusions, and sharing findings for verification by peers.",
    
    "ancient egyptian civilization": "Ancient Egyptian civilization flourished along the Nile River from about 3100 BCE to 332 BCE, developing advanced architecture, writing (hieroglyphics), religion, art, and governmental systems.",
    "world war 2 overview": "World War II (1939-1945) involved Allies (including Britain, France, USSR, USA) against Axis powers (Nazi Germany, Italy, Japan), resulting in 70-85 million casualties and reshaping global politics.",
    "renaissance period": "The Renaissance (14th-17th centuries) was a European cultural movement emphasizing art, science, and learning, inspired by classical antiquity and marked by innovations from figures like Leonardo da Vinci and Michelangelo.",
    "industrial revolution": "The Industrial Revolution (late 1700s-1800s) transformed manufacturing through mechanization, steam power, and factory systems, dramatically changing economic and social structures across Europe and North America.",
    "civil rights movement": "The American Civil Rights Movement (1950s-1960s) fought racial segregation and discrimination through nonviolent protest, legal challenges, and advocacy, led by figures including Martin Luther King Jr. and Rosa Parks."
}

# Create a pool of generic incorrect answers that can be used for any query
incorrect_answers = [
    "The Eiffel Tower is located in Paris and attracts millions of visitors each year.",
    "NBA Finals games are usually played in June and decide the champion of the basketball season.",
    "Chocolate chip cookies are typically made from flour, butter, sugar, and chocolate chips baked until golden.",
    "A combustion engine converts the chemical energy of fuel into mechanical energy for movement.",
    "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.",
    "Chess is a board game played between two players with 16 pieces each on a checkered board of 64 squares.",
    "Photosynthesis is the process by which green plants use sunlight to synthesize foods with carbon dioxide and water.",
    "The Great Wall of China was built over many centuries to protect Chinese states and empires from nomadic invasions.",
    "Jupiter is the largest planet in our solar system, with a mass two and a half times that of all other planets combined.",
    "The human skeletal system consists of 206 bones that provide structure, protect organs, and enable movement.",
    "Shakespeare wrote approximately 37 plays and 154 sonnets during the late 16th and early 17th centuries.",
    "Ballet is a type of performance dance that originated during the Italian Renaissance in the 15th century.",
    "The internet began as ARPANET in the late 1960s, initially connecting computer networks at several universities.",
    "Bees communicate with other bees by dancing to indicate the direction and distance to flowers with good nectar.",
    "The Amazon Rainforest produces about 20% of Earth's oxygen and is home to 10% of the world's known species.",
    "Antibiotics are medications that kill or slow the growth of bacteria but are ineffective against viral infections.",
    "Vincent van Gogh painted 'Starry Night' in 1889 while staying at an asylum in Saint-Rémy-de-Provence, France.",
    "Recycling reduces waste sent to landfills, conserves natural resources, and reduces pollution from production of new materials.",
    "The Olympic flag's five rings represent the five continents and the meeting of athletes from throughout the world.",
    "Tornado Alley refers to the region in the central United States where tornadoes are most frequent.",
    "Coral reefs are among the most diverse ecosystems on Earth, housing approximately 25% of marine species.",
    "The Rubik's Cube is a 3D combination puzzle invented in 1974 by Hungarian sculptor and professor Ernő Rubik.",
    "Diamonds are formed from carbon subjected to high pressure and temperature deep within Earth's mantle.",
    "Coffee is one of the world's most traded commodities, second only to oil in terms of global trade value.",
    "Galileo Galilei's observations with his telescope provided evidence for the Copernican heliocentric theory.",
    "International Space Station orbits Earth at an altitude of approximately 408 kilometers.",
    "The Great Barrier Reef is the world's largest coral reef system, stretching over 2,300 kilometers off Australia's coast.",
    "Social media platforms use algorithms to determine which content to display based on user behavior and preferences.",
    "Origami is the Japanese art of paper folding, transforming a flat sheet into a finished sculpture using folding techniques.",
    "Impressionism began in the 1860s when a group of painters including Monet and Renoir focused on capturing light and movement.",
    "DNA sequencing determines the precise order of nucleotides (A, G, C, T) within a DNA molecule.",
    "Venice is built on a group of 118 small islands separated by canals and linked by over 400 bridges.",
    "Electric vehicles store energy in rechargeable batteries and use electric motors for propulsion instead of internal combustion engines.",
    "Honeybees live in colonies that can contain up to 60,000 bees during peak summer months.",
    "Blockchain technology creates a distributed ledger across a network of computers that validates and records transactions.",
    "The human brain contains approximately 86 billion neurons that communicate through electrical and chemical signals.",
    "The Sahara Desert covers about 3.6 million square miles, making it the largest hot desert in the world.",
    "Mercury is the smallest and innermost planet in the Solar System, completing an orbit around the Sun every 88 Earth days.",
    "Virtual reality technology creates a simulated environment that can be similar to or completely different from the real world.",
    "The Wright brothers made the first successful airplane flight in 1903, staying airborne for 12 seconds and covering 120 feet.",
    "Renewable energy sources include solar, wind, hydroelectric, geothermal, and biomass, derived from naturally replenishing resources.",
    "The Louvre Museum in Paris houses nearly 38,000 objects from prehistory to the 21st century exhibited over an area of 72,735 square meters.",
    "Microprocessors contain millions or billions of tiny transistors on a single chip that process digital information.",
    "Octopuses have three hearts, nine brains, and blue blood, with remarkable problem-solving abilities and camouflage techniques.",
    "Quantum computers use quantum bits or qubits that can exist in multiple states simultaneously, potentially solving certain problems faster than classical computers.",
    "The Mariana Trench is the deepest oceanic trench on Earth, reaching a maximum depth of about 36,070 feet (10,994 meters).",
    "Sign language is a visual language that uses hand shapes, facial expressions, and body movements to communicate.",
    "Hibernation is a state of minimal activity and metabolic depression in endotherms to conserve energy during winter.",
    "The Hubble Space Telescope was launched into low Earth orbit in 1990 and remains in operation, capturing detailed visible-light images.",
    "Volcanic eruptions occur when magma from within Earth's upper mantle rises and breaks through the crust, releasing lava, ash, and gases."
]

# Create the dataset
dataset = []

# Add the existing pairs from the original dataset
existing_pairs = [
    ["what is diabetes", "Diabetes is a chronic disease that occurs when the pancreas cannot produce enough insulin or when the body cannot effectively use the insulin it produces.", 1],
    ["what is diabetes", "The Eiffel Tower is located in Paris and attracts millions of visitors each year.", 0],
    ["python list comprehension", "In Python, a list comprehension is a concise way to create lists using a single line of code.", 1],
    ["python list comprehension", "NBA Finals games are usually played in June and decide the champion of the basketball season.", 0],
    ["weather in london", "London has a temperate maritime climate with mild summers and cool winters, and rainfall distributed fairly evenly throughout the year.", 1],
    ["weather in london", "Chocolate chip cookies are typically made from flour, butter, sugar, and chocolate chips baked until golden.", 0],
    ["benefits of meditation", "Regular meditation practice can reduce stress, enhance concentration, and improve overall emotional well‑being.", 1],
    ["benefits of meditation", "A combustion engine converts the chemical energy of fuel into mechanical energy for movement.", 0]
]

dataset.extend(existing_pairs)

# Calculate how many more pairs we need
remaining_pairs = 100 - len(dataset)
remaining_positive = 50 - sum(1 for _, _, label in dataset if label == 1)
remaining_negative = 50 - sum(1 for _, _, label in dataset if label == 0)

# Create a list of all queries from our defined topics
all_queries = []
for topic_queries in queries.values():
    all_queries.extend(topic_queries)

# Remove queries that are already in the dataset
used_queries = set(query for query, _, _ in dataset)
available_queries = [q for q in all_queries if q not in used_queries]

# Randomly sample from available queries
selected_queries = random.sample(available_queries, min(len(available_queries), max(remaining_positive, remaining_negative)))

# Add positive pairs
for query in selected_queries:
    if remaining_positive <= 0:
        break
    
    if query in relevant_answers:
        dataset.append([query, relevant_answers[query], 1])
        remaining_positive -= 1

# Add negative pairs
for query in selected_queries:
    if remaining_negative <= 0:
        break
    
    # Get incorrect answers excluding any that might match the query topic
    incorrect = random.choice(incorrect_answers)
    dataset.append([query, incorrect, 0])
    remaining_negative -= 1

# If we still need more pairs, create additional pairs with existing queries
if remaining_positive > 0 or remaining_negative > 0:
    all_existing_queries = list(set(query for query, _, _ in dataset))
    
    # Add remaining positive pairs
    while remaining_positive > 0 and len(relevant_answers) > 0:
        query = random.choice(all_existing_queries)
        if query in relevant_answers:
            answer = relevant_answers[query]
            # Check if this exact pair already exists
            if [query, answer, 1] not in dataset:
                dataset.append([query, answer, 1])
                remaining_positive -= 1
    
    # Add remaining negative pairs
    while remaining_negative > 0:
        query = random.choice(all_existing_queries)
        incorrect = random.choice(incorrect_answers)
        # Check if this exact pair already exists
        if [query, incorrect, 0] not in dataset:
            dataset.append([query, incorrect, 0])
            remaining_negative -= 1

# Shuffle the dataset
random.shuffle(dataset)

# Write to pairs.tsv
with open('pairs.tsv', 'w') as f:
    for query, document, label in dataset:
        f.write(f"{query}\t{document}\t{label}\n")

print(f"Generated {len(dataset)} pairs:")
print(f"Positive pairs: {sum(1 for _, _, label in dataset if label == 1)}")
print(f"Negative pairs: {sum(1 for _, _, label in dataset if label == 0)}") 