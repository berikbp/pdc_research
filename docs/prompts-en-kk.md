# The 20 coherent FLORES prompts — English and Kazakh

This replaces the earlier mixed-article selection. Each passage contains four consecutive source entries from ONE article, and the same sentence IDs are used for all four languages. Selection is curated, not a random sample. Source spelling and translations are preserved unchanged. Topical coherence has been reviewed in English and Kazakh; this is not a certification of perfect translation or factual accuracy.

These are text-continuation inputs for an inference benchmark, not question-answering instructions. There are 20 passages, 80 language inputs and 960 measured trials in the current plan. No benchmark has been started.

Selection: `configs/workloads/coherent-dev-v1.json`. Source: https://github.com/facebookresearch/flores/blob/main/flores200/README.md

## Prompt 01 — Triceratops feeding / Трицератопстың қоректенуі

Sentence IDs: 369, 370, 371, 372

Source article: https://en.wikibooks.org/wiki/Wikijunior:Dinosaurs/Triceratops

**English**

A triceratops' teeth would have been able to crush not only leaves but even very tough branches and roots.
Some scientists think Triceratops ate cycads, which are a type of plant that was common in the Cretaceous.
These plants look like a small palm tree with a crown of sharp, spiky leaves.
A Triceratops could have used its strong beak to strip off the leaves before eating the trunk.

**Қазақша**

Трицератопстың тістері жапырақтарды ғана емес, тіпті өте қатты бұтақтар мен тамырларды да сындыра алатын.
Кейбір ғалымдар Трицератоптар бор кезеңінде кең таралған өсімдік түрі саговникті жеді деп есептейді.
Бұл өсімдіктер өткір, үшкір жапырақтардан тұратын басы бар шағын пальма ағашына ұқсайды.
Трицератопс жеуден бұрын жапырақтарды жұлып алу үшін күшті тұмсығын қолдана алатын болған.

## Prompt 02 — Cells and microscopy / Жасушалар және микроскоп

Sentence IDs: 596, 597, 598, 599

Source article: https://en.wikibooks.org/wiki/Wikijunior:Biology/Cells

**English**

A cell is the smallest structural and functional unit of a living (things) organism.
Cell comes from the Latin word cella which means small room.
If you look at living things under a microscope, you will see that they are made of small squares or balls.
Robert Hooke, a biologist from England, saw small squares in cork with a microscope.

**Қазақша**

Жасуша – тірі организмнің ең кіші құрылымдық және функционалдық бірлігі болып табылады.
Жасуша "cella" деген латын сөзінен шыққан, бұл «кішкентай бөлме» дегенді білдіреді.
Тірі ағзаларға микроскоп арқылы қарасаңыз, олардың шағын шаршылардан немесе шариктерден жасалғанын көресіз.
Роберт Хук, Англиядан шыққан биолог, микроскоп арқылы тығында шағын шаршыларды көрді.

## Prompt 03 — States of matter / Заттың агрегаттық күйлері

Sentence IDs: 601, 602, 603, 604

Source article: https://en.wikibooks.org/wiki/Wikijunior:The_Elements/Matter_Is_The_Stuff_Around_You

**English**

Elements and compounds can move from one state to another and not change.
Nitrogen as a gas still has the same properties as liquid nitrogen. The liquid state is denser but the molecules are still the same.
Water is another example. The compound water is made up of two hydrogen atoms and one oxygen atom.
It has the same molecular structure whether it is a gas, liquid, or solid.

**Қазақша**

Элементтер мен қосылыстар бір күйден екінші күйге өтіп, өзгермеуі мүмкін.
Газ түрінде азоттың сипаттары әлі де сұйық азотпен бірдей болады. Сұйық күй тығыздау болады, бірақ молекулалар әлі де бірдей болады.
Су тағы бір мысал. Су екі сутек және бір оттек атомынан тұрады.
Ол газ, сұйықтық немесе қатты зат болсын, оның молекулалық құрылымы бірдей болады.

## Prompt 04 — Understanding time / Уақыт ұғымы

Sentence IDs: 606, 607, 608, 609

Source article: https://en.wikibooks.org/wiki/Wikijunior:How_Things_Work/Time

**English**

Time is something that is all around us, and affects everything we do, yet is difficult to understand.
Time has been studied by religious, philosophical, and scientific scholars for thousands of years.
We experience time as a series of events passing from the future through the present to the past.
Time is also how we compare the duration (length) of events.

**Қазақша**

Уақыт айналамызда барлық жерде бар нәрсе және біз істейтін барлық нәрсеге әсер етеді, дегенмен оны түсіну қиын.
Уақытты мыңдаған жылдар бойы діндарлар, философтар және ғалымдар зерттеп келген.
Біз уақытты болашақтан осы шақ арқылы өткен шаққа өтетін оқиғалардың сериясы ретінде бастан өткереміз.
Уақыт түсінігі оқиғалардың ұзақтығын қалай салыстыратынымызға да байланысты.

## Prompt 05 — Convergent thinking / Конвергенциялық ойлау

Sentence IDs: 625, 626, 627, 628

Source article: https://en.wikibooks.org/wiki/Cognitive_Psychology_and_Cognitive_Neuroscience/Problem_Solving_from_an_Evolutionary_Perspective

**English**

Convergent thinking patterns are problem solving techniques that unite different ideas or fields to find a solution.
The focus of this mindset is speed, logic and accuracy, also identification of facts, reapplying existing techniques, gathering information.
The most important factor of this mindset is: there is only one correct answer. You only think of two answers, namely right or wrong.
This type of thinking is associated with certain science or standard procedures.

**Қазақша**

Конвергенциялық ойлау үлгілері шешімді табу үшін әртүрлі идеяларды немесе салаларды біріктіретін мәселе шешу әдістері болып табылады.
Бұл ойлау үлгісі жылдамдыққа, логикаға және дәлдікке, сондай-ақ фактілерді анықтауға, бар әдістерді қайта қолдануға, ақпаратты жинауға баса көңіл бөледі.
Бұл пікірдің ең маңызды факторы: тек бір дұрыс жауабы бар. Сіз тек екі жауап, дұрыс немесе қате жауап туралы ойлайсыз.
Ойлаудың осы түрі белгілі бір ғылыммен немесе стандартты рәсімдермен байланысты.

## Prompt 06 — Asynchronous learning / Асинхронды оқыту

Sentence IDs: 640, 641, 642, 643

Source article: https://en.wikibooks.org/wiki/Blended_Learning_in_K-12/Synchronous_and_asynchronous_communication_methods

**English**

Asynchronous communication encourages time for reflection and reaction to others.
It allows students the ability to work at their own pace and control the pace of instructional information.
In addition, there are fewer time restrictions with the possibility of flexible working hours. (Bremer, 1998)
The use of the Internet and the World Wide Web allows learners to have access to information at all times.

**Қазақша**

Асинхронды коммуникация ойлану және басқаларға реакция көрсетуге уақыт береді.
Ол студенттерге өз жылдамдығымен жұмыс істеуге және оқу ақпаратының жылдамдығын бақылауға мүмкіндік береді.
Бұған қоса икемді жұмыс уақыты мүмкіндігі арқылы уақыт шектеулері азырақ болады. (Бремер, 1998)
Интернетті және дүниежүзілік желіні пайдалану оқушыларға әрқашан ақпараттың қолжетімділігін қамтамасыз етеді.

## Prompt 07 — Submarines / Сүңгуір қайықтар

Sentence IDs: 402, 403, 404, 405

Source article: https://en.wikibooks.org/wiki/Wikijunior:World_War_II/Submarines

**English**

Submarines are ships designed to travel underwater, and remain there for an extended amount of time.
Submarines were used in World War I and World War II. Back then they were very slow and had a very limited shooting range.
In the beginning of the war they mostly travelled on top of the sea, but as radar began developing and becoming more accurate the submarines were forced to go under water to avoid being seen.
German submarines were called U-Boats. The Germans were very good at navigating and operating their submarines.

**Қазақша**

Субмариналар - бұл су астында жүзетін кемелер және олар ұзақ уақыт су астында қала алады.
Сүңгуір қайықтар Бірінші дүниежүзілік соғыста және Екінші дүниежүзілік соғыста пайдаланылған. Ол кезде сүңгуір қайықтар өте бау болған және ату ауқымы өте шектеулі болатын.
Соғыстың басында олар негізінен теңіздің үстінде жүзетін, бірақ радар дами келе және дәлірек бола келе, сүңгуір қайықтар көрінбеу үшін су астына түсуге мәжбүр болды.
Неміс сүңгуір қайықтары «U кемелері» деп аталатын. Немістер сүңгуір қайықтар навигациясында және оларды басқаруда өте шебер еді.

## Prompt 08 — History of Luxembourg / Люксембург тарихы

Sentence IDs: 433, 434, 435, 436

Source article: https://en.wikibooks.org/wiki/Wikijunior:Europe/Luxembourg

**English**

Luxembourg has a long history but its independence dates from 1839.
Present-day parts of Belgium were part of Luxembourg in the past but became Belgian after the 1830s Belgian Revolution.
Luxembourg has always tried to remain a neutral country but it was occupied in both World War I and World War II by Germany.
In 1957 Luxembourg became a founding member of the organization which is today known as the European Union.

**Қазақша**

Люксембургтің ұзақ тарихы бар, бірақ оның тәуелсіздігі 1839 жылдан басталады.
Бельгияның бүгінгі күнгі бөліктері бұрын Люксембургтың бөлігі болған, бірақ 1830 жылғы бельгиялық революциядан кейін бельгиялық болды.
Люксембург әрдайым бейтарап ел болып қалуға тырысты, бірақ оны бірінші дүниежүзілік соғыста да, екінші дүниежүзілік соғысда Германия басып алды.
1957 жылы Люксембург ұйымның негізін қалаушы мүшеге айналды. Бүгінгі күні бұл Еуропалық Одақ ретінде белгілі. 

## Prompt 09 — Red tide and marine algae / Қызыл толқын және теңіз балдырлары

Sentence IDs: 135, 136, 137, 138

Source article: https://en.wikinews.org/wiki/Red_tide_affects_South_Padre_Island,_Mexico

**English**

Red tide is caused by a higher than normal concentration of Karenia brevis, a naturally-occurring single-celled marine organism.
Natural factors can intersect to produce ideal conditions, allowing this algae to increase in number dramatically.
The algae produces a neurotoxin that can disable nerves in both humans and fish.
Fish often die because of the high concentrations of the toxin in the waters.

**Қазақша**

Қызыл судың көтерілуін Karenia brevis деп аталатын табиғи жалғыз жасушалы теңіз ағзасының қалыптыдан тыс концентрациясы тудырады.
Табиғи факторлар өзара қақтығысып, балдырлардың қарқынды көбеюіне тамаша жағдай жасауы мүмкін.
Балдырлар адамдарда да, балықта да нервтерді бұзуы мүмкін нейротоксин шығарады.
Балық көбінесе сулардағы токсиндердің жоғары концентрацияларына байланысты өледі.

## Prompt 10 — Tropical Storm Danielle / Даниэль тропикалық дауылы

Sentence IDs: 324, 325, 326, 327

Source article: https://en.wikinews.org/wiki/Tropical_Storm_Danielle_forms_in_Atlantic_Ocean

**English**

Tropical Storm Danielle, fourth named storm of the 2010 Atlantic hurricane season, has formed in the eastern Atlantic Ocean.
The storm, located approximately 3,000 miles from Miami, Florida, has maximum sustained winds of 40 mph (64 kph).
Scientists at the National Hurricane Center predict that Danielle will strengthen to a hurricane by Wednesday.
As the storm is far from landfall, it remains difficult to assess potential impact to the United States or Caribbean.

**Қазақша**

«Даниэль» тропикалық дауылы, 2010 жылғы атлантикалық дауылдар маусымындағы төртінші атауы бар дауыл, шығыс Атлант мұхитында қалыптасты.
Флориданың Майами қаласынан шамамен 3 000 миль қашықтықта орналасқан дауылдың максималды тұрақты жылдамдығы 40 миль/сағ (64 км/сағ) құрайды.
Ұлттық дауылдар орталығындағы ғалымдар «Даниэль» сәрсенбі күніне қарай дауылға дейін күшейеді деп болжады.
Дауыл жағадан алыс, сондықтан Америка Құрама Штаттарына немесе Кариб аймағына болуы ықтимал әсерді болжау қиын болып тұр. 

## Prompt 11 — San Francisco economy and tourism / Сан-Франциско экономикасы және туризмі

Sentence IDs: 660, 661, 662, 663

Source article: https://en.wikivoyage.org/wiki/San_Francisco

**English**

While San Francisco's economy is linked to it being a world-class tourist attraction, its economy is diversified.
The largest employment sectors are professional services, government, finance, trade, and tourism.
Its frequent portrayal in music, films, literature and popular culture has helped make the city and its landmarks known throughout the world.
San Francisco has developed a large tourist infrastructure with numerous hotels, restaurants, and top-notch convention facilities.

**Қазақша**

Сан-Франциско экономикасы әлемдік деңгейдегі туристік көрнекі жер болуымен байланысты болса да, оның экономикасы әртараптандырылған.
Ең ірі жұмыспен қамту секторларына кәсіби қызметтер, үкімет, қаржы, сауда және туризм кіреді.
Оның музыка, кино, әдебиет және танымал мәдениеттегі бейнелері, қаланың және ондағы әйгілі орындардың әлемге белгілі болуына көмектесті.
Сан-Франциско қаласында көптеген қонақ үйлер, мейрамханалар және бірінші дәрежелі конференц-залдары бар үлкен туристік инфрақұрылым дамыған.

## Prompt 12 — Grand Canyon camping permits / Гранд-Каньонда кемпингке рұқсат алу

Sentence IDs: 670, 671, 672, 673

Source article: https://en.wikivoyage.org/wiki/Grand_Canyon

**English**

Any camping below the rim in Grand Canyon requires a backcountry permit.
Permits are limited to protect the canyon, and become available on the 1st day of the month, four months prior to the start month.
Thus, a backcountry permit for any start date in May becomes available on 1 Jan.
Space for the most popular areas, such as the Bright Angel Campground adjacent to Phantom Ranch, generally fill up by the requests received on first date they are opened to reservations.

**Қазақша**

Гранд Каньон жиегінен төмен кез келген кемпинг үшін штаттың алыс аудандарына баруға рұқсат алу қажет.
Рұқсаттар каньонды қорғау мақсатында шектелген және олар айдың 1-ші күнінде, басталу айынан төрт ай бұрын қолжетімді болады.
Осылайша мамырдың кез келген басталу күніндегі штаттың шалғай аймақтарына баруға рұқсат 1 қаңтарда қолжетімді болады.
Фантом ранчосына іргелес Брайт Эйнджел лагері сияқты ең танымал аумақтар үшін орын әдетте олар брондауға ашылған бірінші күнде алынған сұраулармен толтырылады.

## Prompt 13 — Victoria Falls / Виктория сарқырамасы

Sentence IDs: 680, 681, 682, 683

Source article: https://en.wikivoyage.org/wiki/Victoria_Falls

**English**

Victoria Falls is a town in the western portion of Zimbabwe, across the border from Livingstone, Zambia, and near Botswana.
The town lies immediately next to the falls, and they are the major attraction, but this popular tourist destination offers both adventure seekers and sightseers plenty of opportunities for a longer stay.
In the rainy season (November to March), the water volume will be higher and the Falls will be more dramatic.
You are guaranteed to get wet if you cross the bridge or walk along the trails winding near the Falls.

**Қазақша**

Виктория Фоллс Зимбабвенің батыс бөлігіндегі, Замбия, Ливингстон шекарасы бойындағы және Ботсвана жанындағы қала болып табылады.
Бұл қала сарқыраманың тура жанында орналасқан және ол негізгі көрнекі жер болып табылады, бірақ бұл танымал туристік жер хикая іздеушілерге де, көрнекі жерді көрушілерге де ұзағырақ қалу үшін көптеген мүмкіндіктерді ұсынады.
Жаңбырлы маусымда (қараша-наурыз) судың көлемі көбірек болады, ал сарқырамалар әсерлірек болады.
Көпір арқылы өтсеңіз немесе сарқырама жанындағы іздер бойымен жүрсеңіз, үстіңіз міндетті түрде су болады.

## Prompt 14 — Visiting the Galapagos by boat / Галапагос аралдарына кемемен бару

Sentence IDs: 712, 713, 714, 715

Source article: https://en.wikivoyage.org/wiki/Galapagos_Islands

**English**

Seeing the sites and wildlife of the Galapagos is best done by boat, just as Charles Darwin did it in 1835.
Over 60 cruise ships ply the Galapagos waters - ranging in size from 8 to 100 passengers.
Most people book their place well in advance (as the boats are usually full during the high season).
Be sure the agent through whom you book is a Galapagos specialist with a good knowledge of a wide variety of ships.

**Қазақша**

Галапагос аралдарының көрнекті жерлерін және жабайы табиғатын қайықпен көрген жөн, 1835 жылы Чарльз Дарвин дәл осылай істеген.
60-тан астам круиздік кемелер Галапагос суларын өтеді, олардың сыйымдылығы 8-ден 100 жолаушыға дейін.
Адамдардың көпшілігі орындарын алдын ала брондайды (өйткені едетте сұраныс жоғары маусымда қайықтар толы болады).
Брон жасайтын агентіңіз көптеген кеме түрлерін білетін Галапагос аралдары бойынша маман екеніне көз жеткізіңіз.

## Prompt 15 — Hospitality exchange networks / Қонақжайлылықпен алмасу желілері

Sentence IDs: 760, 761, 762, 763

Source article: https://en.wikivoyage.org/wiki/Hospitality_exchange

**English**

A hospitality exchange network is the organization that connects travelers with locals in the cities they are going to visit.
Joining such a network usually just requires filling out an online form; although some networks offer or require additional verification.
A listing of available hosts is then provided either in print and/or online, sometimes with references and reviews by other travelers.
Couchsurfing was founded in January 2004 after computer programmer Casey Fenton found a cheap flight to Iceland but did not have a place to stay.

**Қазақша**

Қонақжайлылықпен алмасу желісі – саяхатшыларды өздері баратын қалалардағы жергілікті тұрғындармен байланыстыратын ұйым.
Мұндай желіге қосылу әдетте онлайн пішінді толтыруды қажет етеді; дегенмен, кейбір желілер қосымша растауды ұсынады немесе талап етеді.
Содан кейін қолжетімді қонақүй иелерінің тізімі басылған немесе онлайн түрде, кейде басқа саяхатшылардың сілтемелері мен шолуларымен бірге қамтамасыз етіледі.
Couchsurfing компаниясының негізі 2004 жылдың қаңтар айында, компьютер бағдарламашысы Кейси Фентон Исландияға арзан рейсті тауып, бірақ қонатын жерде таппағанда қаланды.

## Prompt 16 — World fairs and horticultural exhibitions / Дүниежүзілік және бау-бақша көрмелері

Sentence IDs: 773, 774, 775, 776

Source article: https://en.wikivoyage.org/wiki/World%27s_Fair

**English**

A World's Fair (commonly called World Exposition, or simply Expo) is large international festival of arts and sciences.
Participating countries present artistic and educational displays in national pavilions to showcase world issues or their country's culture and history.
International Horticultural Expositions are specialised events which showcase floral displays, botanical gardens and anything else to do with plants.
Although in theory they can take place annually (so long as they are in different countries), in practice they are not.

**Қазақша**

Дүниежүзілік жәрмеңке (көбінесе Дүниежүзілік көрме немесе жай Экспо деп аталады) — үлкен халықаралық өнер және ғылымдар фестивалі.
Қатысушы елдер әлемдік проблемаларды немесе өз елінің мәдениеті мен тарихын көрсету үшін ұлттық павильондарда өнер және білім беру экспозицияларын ұсынады.
Халықаралық бау-бақша көрмелері - бұл гүлдер көрмесін, ботаникалық бақтарды және өсімдіктерге қатысты кез-келген нәрсені көрсететін мамандандырылған іс-шаралар.
Теория жүзінде олар жыл сайын өтуі мүмкін болғанымен (олар әртүрлі елдерде болса), тәжірибе жүзінде олай емес.

## Prompt 17 — Wildlife photography / Жабайы табиғатты суретке түсіру

Sentence IDs: 782, 783, 784, 785

Source article: https://en.wikivoyage.org/wiki/Wildlife_photography

**English**

Wildlife is among the most challenging motifs for a photographer, and needs a combination of good luck, patience, experience and good equipment.
Wildlife photography is often taken for granted, but like photography in general, a picture is worth a thousand words.
Wildlife photography often requires a long telephoto lens, though things like a flock of birds or a tiny creature need other lenses.
Many exotic animals are hard to find, and parks sometimes have rules about taking photographs for commercial purposes.

**Қазақша**

Жабайы табиғат – фотографтардың ең жоғары ынталандыратын сарыны болып табылады, оған сәттілік, сабырлылық, тәжірибе және сапалы жабдық қажет.
Көбінесе жабайы табиғат фотографиясы өзінен-өзі белгілі нәрсе деп есептеледі, бірақ жалпы алғандағы фотография секілді, мың сөзді естігеннен бір суретті көрген жөн. 
Жабайы табиғат фотосуреттерін түсіру көбінесе ұзын телефото объективті қажет етеді, дегенмен құстардың тобы немесе шағын жәндік сияқты нәрселер үшін басқа объективтер керек.
Көп экзотикалық жануарларды табу қиын, ал саябақтарда кейде коммерциялық мақсатта фотосуреттер түсіру туралы ережелер бар.

## Prompt 18 — Day hiking / Бір күндік жаяу серуен

Sentence IDs: 812, 813, 814, 815

Source article: https://en.wikivoyage.org/wiki/Hiking

**English**

Hiking is an outdoor activity which consists of walking in natural environments, often on hiking trails.
Day hiking involves distances of less than a mile up to longer distances that can be covered in a single day.
For a day hike along an easy trail little preparations are needed, and any moderately fit person can enjoy them.
Families with small children may need more preparations, but a day outdoors is easily possible even with babies and pre-school children.

**Қазақша**

Жаяу серуендеу табиғи орта жағдайында жаяу жүруді қамтитын ашық ауада өткізілетін тынығудың белсенді түрі.
Күндізгі жаяу туризмде бір милядан азырақ қашықтықтардан бір күнде өтуге болатын ұзағырақ қашықтықтарға дейін болады.
Бір күндік жорық үшін көп дайындық қажет емес және кез-келген қарапайым адам мұндай дайындықтарды орындай алады.
Кішкентай балалары бар отбасылар көбірек дайындалуы қажет болуы мүмкін, бірақ күнді сыртта өткізу тіпті сәбилер және мектеп жасына дейінгі балалар бар болса да оңай мүмкін болады.

## Prompt 19 — Text translation applications / Мәтін аударуға арналған қолданбалар

Sentence IDs: 915, 916, 917, 918

Source article: https://en.wikivoyage.org/wiki/Smartphone_apps_for_travellers

**English**

Real-time text translator apps – applications that are capable of automatically translating whole segments of text from one language into another.
Some of the applications in this category can even translate texts in foreign languages on signs or other objects in the real world when the user points the smartphone towards those objects.
The translation engines have improved dramatically, and now often give more or less correct translations (and more seldom gibberish), but some care is due, as they still may have gotten it all wrong.
One of the most prominent apps in this category is Google Translate, which allows offline translation after downloading the desired language data.

**Қазақша**

Нақты уақыттағы мәтіндік аудармашы қолданбалары — мәтіннің бүтіндей сегменттерін бір тілден екінші тілге автоматты түрде аударуға қабілетті қолданбалар.
Осы санаттағы кейбір қосымшалар, пайдаланушы смартфонды белгілі бір нысандарға бағыттаған кезде, белгілерді немесе әлемдегі басқа нақты заттарды, шет тілдеріндегі мәтіндерді аудара алады.
Аударма жүйелері едәуір жақсарды, енді азды-көпті дұрыс аудармаларды жиі береді (және жарамсыз аудармалар сирегірек кездеседі), бірақ бәрібір сақ болу қажет, өйткені олар бәрібір барлығын дұрыс емес түсінуі мүмкін.
Бұл категориядағы ең танымал қолданбалардың бірі – Google Translate, бұл қажетті тіл деректерін жүктеп алғаннан кейін желіден тыс аудару мүмкіндігін береді.

## Prompt 20 — Cinque Terre villages / Чинкве Терре ауылдары

Sentence IDs: 967, 968, 969, 970

Source article: https://en.wikivoyage.org/wiki/Cinque_Terre

**English**

Cinque Terre, which means Five Lands, comprises the five small coastal villages of Riomaggiore, Manarola, Corniglia, Vernazza and Monterosso located in the Italian region of Liguria.
They are listed on the UNESCO World Heritage List.
Over the centuries, people have carefully built terraces on the rugged, steep landscape right up to the cliffs that overlook the sea.
Part of its charm is the lack of visible corporate development. Paths, trains and boats connect the villages, and cars cannot reach them from the outside.

**Қазақша**

Бес жерді білдіретін Чинкве Терре Италияның Лигурия аймағында орналасқан бес кішкентай жағалаудағы Риомаггио, Манарола, Корнилла, Вернацца және Монтероссо ауылдарын қамтиды.
Олар ЮНЕСКО-ның Бүкіләлемдік мұра тізіміне енген.
Ғасырлар бойы, адамдар теңізге қарайтын жартастарға баратын тік пейзажда террасаларды мұқият соғып келген.
Оның сүйкімділігінің бір бөлігі — көрінетін корпоративтік дамудың болмауы. Жолдар, пойыздар және қайықтар ауылдарды жалғастырады, ал көліктер оларға сырттан жете алмайды.
