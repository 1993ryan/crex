import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

rcParams['axes.unicode_minus'] = False
# rcParams['font.sans-serif'] = ['Simhei']
rcParams['font.sans-serif'] = ['Times New Roman']


y1 = [0.9926442503929138, 0.9571090936660767, 0.9522356986999512, 0.6303988695144653, 0.9483774900436401, 0.9464377760887146, 0.9998241662979126, 0.9391101598739624, 0.9991835355758667, 0.9662824869155884, 0.9792293906211853, 0.9630652666091919, 0.9904848337173462, 0.9946126937866211, 0.9773818254470825, 0.9877079725265503, 0.9956605434417725, 0.9993960857391357, 0.988242506980896, 0.9982386827468872, 0.9865335822105408, 0.968412458896637, 0.7859527468681335, 0.9748052954673767, 0.981859028339386, 0.9913372993469238, 0.9449242949485779, 0.9540044665336609, 0.9600513577461243, 0.9663103222846985, 0.999424934387207, 0.9996034502983093, 0.9977543950080872, 0.9937909841537476, 0.9988057613372803, 0.9989911317825317, 0.9658411145210266, 0.970913290977478, 0.9926887154579163, 0.9987936615943909, 0.9898006916046143, 0.9991322755813599, 0.9464181065559387, 0.9572596549987793, 0.9804123640060425, 0.9489032626152039, 0.9983961582183838, 0.9985985159873962, 0.9992892742156982, 0.9993270635604858, 0.9994838237762451, 0.9994708299636841, 0.9993681907653809, 0.999211311340332, 0.9991412162780762, 0.9998773336410522, 0.9900248050689697, 0.9528205990791321, 0.9997409582138062, 0.9497688412666321, 0.9937114715576172, 0.9867403507232666, 0.9981041550636292]
y2 = [0.9931746125221252, 0.9973487854003906, 0.9995037317276001, 0.9995476603507996, 0.9997527599334717, 0.9862297773361206, 0.9991384744644165, 0.9990866780281067, 0.9872444868087769, 0.9977642297744751, 0.9942444562911987, 0.9995570182800293, 0.9986822605133057, 0.992772102355957, 0.9974161386489868, 0.998394787311554, 0.9981892704963684, 0.9943037629127502, 0.9971842765808105, 0.9772933721542358, 0.8602917194366455, -0.0641568973660469, 0.9192341566085815, 0.9990890026092529, 0.998441219329834, 0.9992901086807251, 0.9979536533355713, 0.9987301230430603, 0.994924783706665, 0.9993686676025391, 0.9996170997619629, 0.9995988011360168, 0.9969642162322998, 0.9939069151878357, 0.9992859363555908, 0.9960076808929443, 0.8734530806541443, 0.9718025922775269, 0.9558439254760742, 0.9722722172737122, 0.9988179206848145, 0.9996837973594666, 0.9988632202148438, 0.9995331168174744, 0.9686908721923828, 0.9995389580726624, 0.9914785623550415, 0.9906946420669556, 0.9979993104934692, 0.9991987347602844, 0.9870331883430481, 0.9903267025947571, 0.9976912140846252, 0.9994286298751831, 0.9960703253746033, 0.9990010857582092, 0.9995437264442444, 0.9992848634719849, 0.9986819624900818, 0.9994832277297974, 0.9995546936988831, 0.9983867406845093, 0.9944925308227539, 0.9999058842658997, 0.9936249256134033, 0.7518682479858398, 0.9985936284065247, 0.9967817664146423, 0.9996310472488403, 0.970369815826416, 0.9962942004203796, 0.9992450475692749, 0.9997683763504028, 0.9833547472953796, 0.9995369911193848, 0.9990831017494202, 0.9990476965904236, 0.9993106126785278, 0.9988706707954407, 0.9077616930007935, 0.9995819330215454, 0.9951257705688477, 0.9990047812461853, 0.9784659147262573, 0.9662473797798157, 0.9996856451034546, 0.9929365515708923, 0.9957425594329834, 0.9980307221412659, 0.9972471594810486, 0.9992337822914124, 0.9993782639503479, 0.9987137913703918, 0.9993625283241272, 0.9940896034240723, 0.9929869174957275, 0.9995729327201843, 0.9992704391479492, 0.9994823336601257, 0.9996223449707031, 0.9765874147415161, 0.9992268085479736, 0.9932966828346252, 0.9989620447158813, 0.9874483942985535, 0.9991015195846558, 0.9996926784515381, 0.999125063419342, 0.9981605410575867, 0.9991840124130249, 0.8924375772476196, 0.9965745806694031, 0.9984971284866333, 0.9991602897644043, 0.9955856800079346, 0.9987857341766357, 0.9982905983924866, 0.9943228960037231, 0.9997780919075012, 0.9929354190826416, 0.9990524649620056, 0.9998713135719299, 0.999391496181488, 0.9997943043708801, 0.9768258333206177, 0.9998924732208252, 0.9676896333694458, 0.999808132648468, 0.9998166561126709, 0.9410901665687561, 0.9989782571792603, 0.9959426522254944, 0.9990565776824951, 0.9992838501930237, 0.9998921751976013, 0.9982038140296936, 0.9992976188659668, 0.9990345239639282, 0.9992007613182068, 0.9986451268196106, 0.9996649026870728, 0.9994851350784302, 0.9820394515991211, 0.9480103254318237, 0.9991077184677124, 0.9982483386993408, 0.9988840818405151, 0.9992178678512573, 0.9832662343978882, 0.9993406534194946, 0.9993346929550171, 0.9998837113380432, 0.9967375993728638, 0.9982470273971558, 0.9976325035095215, 0.999617874622345, 0.9992183446884155, 0.9995417594909668, 0.9992671608924866, 0.9789815545082092, 0.9544171094894409, 0.9868961572647095, 0.9418118000030518, 0.9965283870697021, 0.9982830286026001, 0.9983041882514954, 0.9959751963615417, 0.9989730715751648, 0.9994155168533325, 0.9996662735939026, 0.9997245669364929, 0.9995084404945374, 0.9994671940803528, 0.9998343586921692, 0.9997727274894714, 0.9690308570861816, 0.9993113875389099, 0.9997013807296753, 0.9990724921226501, 0.999482274055481, 0.9998138546943665, 0.9346687197685242, 0.9990571141242981, 0.9995322227478027, 0.9844871759414673, 0.999302089214325, 0.9983450174331665, 0.8212313055992126, 0.8338471055030823, 0.9993399381637573, 0.9994773268699646, 0.9992334246635437, 0.9982144832611084, 0.9997205138206482, 0.9997006058692932]
y3 = [0.9946410655975342, 0.999758780002594, 0.9422962069511414, 0.9797398447990417, 0.998324990272522]
y4 = [0.9967830181121826, 0.9995837211608887, 0.9992610216140747]
y5 = [0.9990529417991638, 0.999719500541687, 0.999787449836731, 0.9933351874351501, 0.997640073299408, 0.9991753697395325, 0.9979402422904968, 0.9948295950889587, 0.9996024370193481, 0.9990474581718445, 0.9988071918487549, 0.9997881054878235, 0.9993516802787781, 0.9782103300094604, 0.9998185038566589, 0.999210000038147, 0.9936318397521973, 0.9996142387390137, 0.9982647895812988, 0.9991610050201416, 0.9865338206291199, 0.9962761998176575, 0.9996079802513123, 0.9980317950248718, 0.999214768409729, 0.9939528703689575, 0.9985251426696777, 0.8854401707649231, 0.9993576407432556, 0.999461829662323, 0.999762773513794, 0.9993997812271118, 0.9994598627090454, 0.9982634782791138, 0.989314079284668, 0.9997044205665588, 0.9957064390182495, 0.9998761415481567, 0.6901909708976746, 0.9981985688209534, 0.9993501305580139, 0.9979546070098877, 0.995224118232727, 0.9248293042182922, 0.9991326332092285, 0.9992941617965698, 0.9979386925697327, 0.9971886873245239, 0.9997388124465942, 0.9993179440498352, 0.9906538724899292, 0.9989268183708191, 0.981597363948822, 0.9873183369636536, 0.9980149865150452, 0.9977055788040161, 0.9840425252914429, 0.998696506023407, 0.9628797173500061, 0.9993310570716858, 0.9994140863418579, 0.9984052181243896, 0.9995582699775696, 0.9995195865631104, 0.9971116781234741, 0.9994536638259888, 0.9997397661209106, 0.9988497495651245, 0.9992913603782654, 0.9550220966339111, 0.9995554685592651, 0.9991931915283203, 0.9997665286064148, 0.9987863302230835, 0.9974454641342163, 0.9997783303260803, 0.986315131187439, 0.9980497360229492, 0.9996936321258545, 0.9990041255950928]
y6 = [0.9980422258377075, 0.959439754486084, 0.9994257092475891, 0.9987776875495911, 0.9987545609474182, 0.9812893867492676, 0.9929850101470947, 0.9921319484710693, 0.9974249005317688, 0.9994621276855469, 0.9860603213310242, 0.9998095035552979, 0.9992533326148987, 0.9992095232009888, 0.994460940361023, 0.9997074604034424, 0.9998854398727417, 0.9998790621757507, 0.9998287558555603, 0.9996688961982727, 0.9995080828666687, 0.9975264072418213, 0.9995497465133667, 0.9972028732299805, 0.9982481002807617, 0.9994385242462158, 0.9725570678710938, 0.991516649723053, 0.9996626377105713, 0.9993151426315308, 0.864686131477356, 0.9994229674339294, 0.9997509121894836, 0.99358069896698, 0.9943100810050964, 0.9997396469116211, 0.9977705478668213, 0.9995541572570801]
y7 = [0.9360281825065613, 0.999420166015625, 0.9902506470680237, 0.45564502477645874, 0.9788142442703247, 0.9377006888389587, 0.9650593996047974, 0.9891080856323242, 0.9874433279037476, 0.9990687966346741, 0.9471747875213623, 0.9991832375526428, 0.992542564868927, 0.9990894198417664]
y8 = [0.9951013326644897, 0.9994722008705139, 0.9689974188804626, 0.9993730783462524, 0.9956849813461304, 0.9975364804267883, 0.9987008571624756, 0.8454577922821045]
y9 = [0.9954700469970703, 0.9960939884185791, 0.9742860794067383, 0.9969825744628906, 0.9970084428787231]
y10 = [0.9851464033126831, 0.9967710375785828, 0.9615469574928284, 0.9945186376571655, 0.9880933165550232, 0.9724998474121094, 0.9881888031959534, 0.99778151512146, 0.999557375907898, 0.9960454106330872, 0.9910138249397278, 0.9994392991065979, 0.9867044687271118, 0.9903914332389832, 0.9942528605461121, 0.9923510551452637, 0.9983684420585632, 0.5609145164489746, 0.9903209805488586, 0.9304617047309875, 0.9584838151931763, 0.9964392185211182, 0.9988267421722412, 0.9992814064025879, 0.9996743202209473, 0.9988905191421509, 0.9916642904281616, 0.9823580384254456, 0.9926813840866089, 0.9983565211296082, 0.998760461807251, 0.9979931712150574, 0.9974342584609985, 0.9949695467948914, 0.9988518357276917, 0.9897611141204834, 0.9922022819519043, 0.9810315370559692, 0.9979612827301025, 0.9990608096122742, 0.9972613453865051, 0.9957252144813538, 0.9971643686294556, 0.9994807243347168, 0.9997814893722534, 0.998725414276123, 0.9896582961082458, 0.9678716063499451, 0.9988795518875122, 0.9882622957229614, 0.9992536306381226, 0.9807271957397461, 0.9741429686546326, 0.9701777696609497, 0.9975489974021912, 0.9598066806793213, 0.998324453830719]
y11 = [0.9988248944282532, 0.9986535906791687, 0.9801549911499023, 0.9969193339347839, 0.9984714984893799, 0.9994955062866211, 0.9777282476425171, 0.9932190775871277, 0.9351793527603149, 0.9838820695877075, 0.9527751207351685, 0.9900959730148315, 0.9647859930992126, 0.9365541934967041, 0.9546818733215332, 0.9982315897941589, 0.9824945330619812, 0.9734845161437988, 0.996003270149231, 0.9603817462921143, 0.9767916202545166, 0.9786557555198669, 0.8987167477607727, 0.9850748181343079, 0.9889945387840271, 0.9751585721969604, 0.9949427247047424, 0.9845679998397827]
y12 = [0.9796309471130371, 0.9941831827163696, 0.9784406423568726, 0.7069323062896729, 0.9986661076545715, 0.9942653775215149, 0.9712188243865967, 0.9617375135421753, 0.969727635383606, 0.9737104177474976, 0.9595246315002441, 0.9297902584075928, 0.9696169495582581, 0.9700497388839722, 0.9498951435089111, 0.9892040491104126, 0.9997518062591553, 0.9928559064865112, -0.2419586032629013, 0.9949105381965637, 0.9883309602737427, 0.9962669610977173, 0.9964971542358398, 0.994006872177124, 0.9921069145202637, 0.9882022142410278, 0.9950476884841919, 0.997143566608429, 0.9966821670532227, 0.9960542321205139, 0.9969847202301025, 0.9700741767883301, -0.13176776468753815, 0.9950025081634521, -0.12403236329555511, 0.999596118927002, 0.9769735336303711, 0.9945141077041626, 0.98688143491745, 0.9968746304512024, 0.9980547428131104, 0.960239589214325, 0.964373767375946, 0.9835821986198425, 0.9840779304504395, 0.9656934142112732, 0.9993633031845093, 0.9919770956039429, 0.9737091064453125, 0.9924474954605103, 0.9813717603683472, 0.9964398741722107, 0.9826960563659668, 0.021735232323408127, 0.9951626062393188, 0.9991745352745056, 0.9915787577629089, 0.9931144118309021, 0.8855383992195129, 0.9955217838287354, 0.9899391531944275, 0.9988585710525513, 0.3948080241680145, 0.9957528114318848, 0.9998327493667603, 0.996549129486084, 0.9962987899780273, 0.9954048991203308, 0.9998332858085632, 0.9894801378250122, 0.9766027331352234, 0.9989340305328369, 0.9612035751342773, 0.8668182492256165, 0.9930275678634644, 0.9822669625282288, 0.9804466962814331, 0.998744010925293, 0.9954637885093689, 0.9979063868522644, 0.8370074033737183, 0.9938541054725647, 0.9901906847953796, 0.999762773513794, 0.9752936363220215, 0.9958319664001465, 0.9874955415725708, 0.9923486113548279, 0.9854015111923218, 0.9916068315505981, 0.996894896030426, 0.9338287115097046, 0.9968106150627136, 0.993637204170227, 0.9967473745346069, 0.9419417381286621, 0.9930245280265808, 0.998925507068634, 0.9742980003356934, 0.9959978461265564, 0.9976706504821777, 0.9869176149368286, 0.998820960521698, 0.9975532293319702, 0.9985533952713013, 0.9477598667144775, 0.9972798824310303, 0.998150646686554, 0.9977670907974243, 0.9881929755210876, 0.9982786178588867, 0.9933491945266724, 0.98687344789505, 0.998293936252594, -0.2055540829896927, 0.9976363182067871, -0.11721649020910263, 0.9879909753799438, 0.9330525398254395, 0.9422422051429749, 0.03243473172187805, 0.9851996898651123, 0.9820200800895691, 0.9855239987373352, 0.34133651852607727, 0.9908549785614014, 0.9793378114700317, 0.9914252758026123, 0.9819881319999695, 0.9809013605117798, 0.9852961897850037, 0.9890430569648743, 0.9866383075714111, 0.9991319179534912, 0.998960554599762, 0.9994466304779053, 0.9937635660171509, 0.9946246147155762, 0.7606750130653381, 0.9309878349304199, 0.9911859035491943, 0.9878131747245789, 0.9533976316452026, 0.21740110218524933, 0.9983646869659424, 0.998443067073822, 0.9995701313018799, 0.9988815784454346, 0.9940869808197021, 0.9905754327774048, 0.9946961402893066, 0.734318196773529, 0.8182827830314636, 0.7275592684745789, 0.9988599419593811, 0.9902287125587463, 0.9887174963951111, 0.9994974136352539, 0.9991645812988281, 0.9988518953323364, 0.9653280377388, 0.9979005455970764, 0.989251971244812, 0.998712956905365, 0.9994100332260132, 0.9877040386199951, 0.995553195476532, 0.9997146725654602, 0.9780041575431824, 0.9969366192817688, 0.9923841953277588, 0.9939813613891602]
y13 = [0.986060380935669, 0.9982007145881653, 0.9963384866714478, 0.9993927478790283, 0.9993757009506226, 0.9449923038482666, 0.9995866417884827, 0.9651263952255249, 0.9837126135826111, 0.9928752183914185, 0.9987790584564209, 0.9991210699081421, 0.9988315105438232, 0.9974050521850586, 0.9983038306236267, 0.9991997480392456, 0.9602707624435425, 0.9984170198440552, 0.9970085620880127, 0.9843696355819702, 0.9947210550308228, 0.9842767715454102, 0.9995318651199341, 0.9993662238121033, 0.9847257137298584, 0.8633277416229248, 0.9372119307518005, 0.9449731111526489, 0.9943546056747437, 0.9679110050201416, 0.9993628859519958, 0.9852257966995239, 0.9940727949142456, 0.9966346025466919, 0.9957887530326843, 0.8528346419334412, 0.9931673407554626, 0.9891576170921326, 0.992449164390564, 0.9885703325271606, 0.9991328716278076, 0.9985376596450806, 0.8551438450813293, 0.9989352226257324, 0.9978336095809937, 0.9993025064468384, 0.9913206100463867, 0.983147144317627, 0.9859061241149902, 0.9929442405700684, 0.8444505333900452, 0.9130586981773376, 0.9981566071510315, 0.9978163242340088, 0.9998393058776855, 0.991409182548523, 0.995086133480072, 0.994111955165863, 0.9996333718299866, 0.9920839071273804, 0.9970828294754028, 0.9945526123046875, 0.9963147640228271, 0.9874621629714966, 0.9457465410232544, 0.9526042938232422, 0.9986584186553955, 0.994641900062561, 0.9862267971038818, 0.9869756102561951, 0.9861027002334595, 0.9996657967567444, 0.9986541271209717, 0.997587263584137, 0.9823769330978394, 0.9933921694755554, 0.994209885597229, 0.9601553082466125, 0.7378749251365662, 0.7393269538879395, 0.9634600877761841, 0.9800611138343811, 0.9899477958679199, 0.9398026466369629, 0.9970723390579224, 0.9953604936599731, 0.9996244311332703, 0.9986788630485535, 0.9861836433410645, 0.9886139035224915, 0.9626688957214355, 0.9917919635772705, 0.9758999347686768, 0.9633775949478149, 0.9987205266952515, 0.9934348464012146, 0.9954538941383362, 0.9852893948554993, 0.9640284776687622, 0.9902928471565247, 0.9994977712631226, 0.9930980205535889, 0.9930587410926819, 0.9687410593032837, 0.9970825910568237, 0.9990283250808716, 0.9986134767532349, 0.9987621307373047, 0.9903113842010498, 0.8938490748405457, 0.9956777095794678, 0.9974331855773926, 0.9975970983505249, 0.993137538433075, 0.999400794506073, 0.8780977129936218, 0.9730270504951477, 0.9569587707519531, 0.9912885427474976, 0.992067277431488, 0.9940506219863892, 0.9314348697662354, 0.9631990790367126, 0.9868184328079224, 0.9908900856971741, 0.9926003217697144, 0.9995542764663696, 0.9997594356536865, 0.9968823790550232, 0.9947213530540466, 0.9921339750289917, 0.9804562926292419, 0.9956024289131165, 0.9684697389602661, 0.9696093797683716, 0.46535810828208923, 0.9990535974502563, 0.9984033107757568, 0.9751336574554443, 0.9978494644165039, 0.9980326890945435]
y14 = [0.9012042284011841, 0.9940585494041443, 0.9877063632011414, 0.9982092976570129, 0.9983084201812744, 0.9996233582496643, 0.9925222992897034, 0.9008305072784424, 0.9995827078819275, 0.9995026588439941, 0.999036431312561, 0.9941748976707458, 0.9961466789245605, 0.9210203886032104, 0.8971274495124817, 0.9893339276313782]
y15 = [0.9978477358818054, 0.9963089823722839, 0.9883696436882019, 0.9972127079963684, 0.997127890586853, 0.9993172883987427, 0.9994688034057617, 0.9458296895027161, 0.9995187520980835, 0.9992533922195435, 0.9994258880615234, 0.996785581111908]
y16 = [0.9984645247459412, 0.9997252225875854, 0.94036865234375, 0.9409781694412231, 0.9109044075012207, 0.9340760707855225, 0.9895681738853455, 0.9908096790313721, 0.9952427744865417, 0.9953054785728455, 0.99491947889328, 0.8466994762420654, 0.9753758907318115, 0.9744759798049927, 0.9770240187644958, 0.9925201535224915, 0.9969375133514404, 0.9924915432929993, 0.9989467263221741, 0.9975458383560181, 0.9901912212371826, 0.9978069067001343, 0.967312753200531, 0.9676743745803833, 0.9973912239074707, 0.981261134147644, 0.983256459236145, 0.9930298924446106, 0.9950736165046692, 0.9864180684089661, 0.9924319982528687, 0.992512047290802, 0.993209183216095, 0.9953650832176208, 0.9920668601989746, 0.9916079640388489, 0.9960530996322632, 0.9973658919334412, 0.9967583417892456, 0.9991296529769897, 0.995219349861145, 0.9974334239959717, 0.9950830936431885, 0.9866071939468384, 0.9983005523681641, 0.586082935333252, 0.9993194937705994, 0.9990284442901611, 0.9989566206932068, 0.9774230718612671, 0.9990560412406921, 0.6373478174209595, 0.991199791431427, 0.9964781999588013, 0.9782477617263794, 0.9366708397865295, 0.9993308782577515, 0.9961717128753662, 0.9918597340583801, 0.9935164451599121, 0.9981549978256226, 0.9919611215591431, 0.9869762659072876, 0.9871176481246948, 0.9920922517776489, 0.9826101064682007, 0.967018723487854, 0.9987569451332092, 0.8974000215530396, 0.9603076577186584, 0.9638856649398804, 0.988197922706604, 0.996123731136322, 0.9374817609786987, 0.9981040954589844, 0.9984064102172852, 0.6310447454452515, 0.9946961402893066, 0.9619115591049194, 0.997544527053833, 0.9858807921409607, 0.9810864329338074, 0.9503679275512695, 0.9585175514221191, 0.9610589146614075, 0.9945908188819885, 0.9953001141548157, 0.7506039142608643, 0.9869754910469055, 0.9988988041877747, 0.9990047812461853, 0.9733883738517761, 0.9924669861793518, 0.9740393161773682, 0.9745397567749023, 0.9983829259872437, 0.9955978393554688, 0.9920647144317627, 0.9841713905334473, 0.9951826333999634, 0.9972601532936096, 0.9930303692817688, 0.9959033131599426]
y17 = [0.9770146012306213, 0.9440816044807434, 0.9898480772972107, 0.9949133396148682, 0.9936343431472778, 0.9798410534858704, 0.9906421303749084, 0.995790421962738]
y18 = [0.8465409874916077, 0.988216757774353, 0.9698678255081177, 0.9958226084709167, 0.9984684586524963, 0.999170184135437, 0.9917029738426208, 0.9909744262695312, 0.9972375631332397, 0.9918907880783081]
y19 = [0.9974925518035889, 0.9968447685241699, 0.9988844394683838, 0.9988363981246948, 0.9997324347496033, 0.9981825947761536, 0.998176634311676, 0.9990310668945312, 0.9984012246131897, 0.9995326995849609, 0.9967904686927795, 0.9994921684265137, 0.9985961318016052, 0.9993222951889038, 0.9992717504501343, 0.9997579455375671, 0.9997052550315857, 0.9976919889450073, 0.9984294176101685, 0.9989935159683228, 0.995229184627533, 0.9995818138122559, 0.9995911717414856, 0.9982680082321167, 0.9983057975769043, 0.9973562955856323, 0.8907878398895264, -0.020563600584864616, 0.9962280988693237, 0.9983292818069458, 0.999536395072937, 0.9995768666267395, 0.9991893768310547, 0.9998259544372559, 0.9990867376327515, 0.9993845820426941, 0.9999010562896729, 0.9992457628250122, 0.9990891218185425, 0.9965817332267761, 0.9998054504394531, 0.9935229420661926, 0.999413788318634, 0.9995388388633728, 0.9933592081069946, 0.9996910691261292, 0.9683720469474792, 0.9992535710334778, 0.999383270740509, 0.9995920062065125, 0.9996098279953003, 0.9992620348930359, 0.9987330436706543, 0.9988768100738525, 0.9998712539672852, 0.9990546107292175, 0.9995425343513489, 0.998773992061615, 0.9977096319198608, 0.9975740909576416, 0.9997037053108215, 0.9997867941856384, 0.9995784759521484, 0.9996433854103088, 0.9994995594024658, 0.9989150762557983]
y20 = [0.9995250701904297, 0.997503936290741, 0.9990367889404297, 0.9996347427368164, 0.9995566010475159, 0.9905612468719482]
y21 = [0.9898130297660828, 0.9696024656295776, 0.9295182228088379]
y22 = [0.9922516345977783, 0.9832032918930054, 0.9935056567192078, 0.9912373423576355, 0.9920545816421509, 0.9898678064346313, 0.9980135560035706, 0.9983646273612976, 0.905326783657074, 0.9905847907066345, 0.9963862895965576, 0.9974254369735718, 0.9978979229927063, 0.9993347525596619, 0.9975769519805908, 0.9837253093719482, 0.9933777451515198, 0.9521973133087158, 0.45517176389694214, 0.9935579299926758, 0.9952684640884399, 0.9913758635520935, 0.9914621710777283, 0.9971382021903992, 0.9868426322937012, 0.9959498643875122, 0.9946606159210205, 0.9990071654319763, 0.9924675226211548, 0.999824047088623, 0.9993554949760437, 0.9915112257003784, 0.983790397644043, 0.7412534952163696, 0.9921230673789978, 0.9876983761787415, 0.9858481884002686, 0.9975653290748596, 0.9993823170661926, 0.9944531321525574, 0.9767013788223267, 0.9503077864646912, 0.839179515838623, 0.997092068195343, 0.964891791343689, 0.993798017501831, 0.9851408004760742, 0.981086015701294, 0.9861456751823425, 0.9943342804908752, 0.9911618828773499, 0.9969874024391174, 0.997925877571106, 0.9979062080383301, 0.9832820296287537, 0.9949972629547119, 0.9948108792304993, 0.9923224449157715, 0.9974223375320435, 0.9884560704231262, 0.9972798228263855, 0.994681715965271, 0.9922432899475098, 0.9998329877853394, 0.9996055364608765, 0.9960508346557617, 0.9848662614822388, 0.9930786490440369, 0.996832013130188, 0.9870060682296753, 0.9952574968338013, 0.9849460124969482, 0.9867770671844482, 0.9884368181228638, 0.999864935874939, 0.8375014662742615, 0.9729158282279968, -0.2536908984184265, 0.9587323665618896, 0.9664103984832764, 0.9766262173652649, 0.9896907210350037, 0.9887996315956116, 0.9857802391052246, 0.9288153648376465, 0.9762308597564697, 0.9739509224891663, -0.19767743349075317, 0.9848048090934753, 0.9993654489517212, 0.9956510066986084, 0.9963177442550659, 0.9955869317054749, 0.9965471625328064, 0.9957267045974731, 0.9884684085845947, 0.3040691614151001, 0.994640052318573, 0.9956461191177368, 0.9966346621513367, 0.9968744516372681, 0.9950947761535645, 0.9972866773605347, 0.9973247051239014, 0.9895823001861572, 0.9987564086914062, 0.9914101362228394, 0.9930936694145203, -0.10592935234308243, 0.9903881549835205, 0.909485936164856, 0.9909474849700928, 0.9698954224586487, 0.9971157312393188, 0.9888683557510376, 0.9997180104255676, 0.988314688205719, 0.9980339407920837, 0.9911701083183289, 0.9523176550865173, 0.9856728315353394, 0.9907528758049011, 0.9893447756767273, 0.9989690184593201, 0.9820605516433716, 0.976803183555603, 0.9711620211601257, 0.9602186679840088, 0.9995375275611877, 0.997215211391449, 0.9812386631965637, 0.9851233959197998, 0.9687021374702454, 0.9908159375190735, 0.9962495565414429, 0.9963012933731079, 0.9995431900024414]
y23 = [0.9949661493301392, 0.5178482532501221, 0.4034098982810974, 0.9914723634719849, 0.5201053023338318]
y24 = [0.993850827217102, 0.9994535446166992, 0.9377827644348145, 0.9594360589981079, 0.9998077750205994]
y25 = [0.9964678287506104, 0.9991598129272461, 0.9894102811813354, 0.9662845134735107, 0.9994872212409973, 0.9988120794296265, 0.987517774105072]
y26 = [0.9804178476333618, 0.9946406483650208, 0.9955220222473145, 0.9982290863990784, 0.9925525188446045]
y27 = [0.997445285320282, 0.9985142946243286, 0.9994193315505981, 0.9971818327903748, 0.9995394349098206, 0.9991909861564636, 0.998553991317749, 0.9997868537902832, 0.9984267354011536, 0.9989511966705322, 0.9995920658111572, 0.9980483055114746, 0.9974520802497864, 0.9975879788398743, 0.9984222650527954, 0.9992395043373108, 0.9981818199157715, 0.9973450899124146, 0.9243180751800537, 0.9984759092330933, 0.9983415007591248, 0.901326596736908, 0.8763432502746582, 0.9982319474220276, 0.9983131885528564, 0.9952386021614075, 0.9995230436325073, 0.9990741014480591, 0.9992637038230896, 0.9949778914451599, 0.9906399846076965, 0.9994165897369385, 0.9893569946289062, 0.9994996190071106, 0.9866750240325928, 0.999829888343811, 0.998542845249176, 0.9932608008384705, 0.9989439845085144, 0.9924592971801758, 0.9994971752166748, 0.9988613724708557, 0.939405620098114, 0.9989712238311768, 0.9410654902458191, 0.9995821118354797, 0.99834144115448, 0.9992460012435913, 0.9998050332069397, 0.9980050921440125, 0.9667928814888, 0.9998018145561218, 0.9981232285499573, 0.5745434165000916, 0.9991414546966553, 0.9995782375335693, 0.9866491556167603, -0.0036265472881495953, 0.9986486434936523, 0.9697070717811584, 0.9993075132369995, 0.9995365738868713, 0.9538949728012085, 0.9937670826911926, 0.9972087740898132, 0.9971871376037598, 0.996975839138031, 0.9990421533584595, 0.9995261430740356, 0.996489405632019, 0.9978730082511902, 0.9971585273742676, 0.9987937211990356, 0.9981138110160828, 0.9989833831787109, 0.9981497526168823, 0.9998493790626526, 0.998993456363678, 0.9991939067840576, 0.9970875978469849, 0.9982923865318298, 0.9989257454872131, 0.9989859461784363, 0.9907439351081848, 0.9965428709983826, 0.9460132122039795, 0.9984166622161865, 0.9579125642776489, 0.999484121799469, 0.9988871216773987, 0.9419070482254028, 0.9976414442062378, 0.9989607930183411, 0.9995144009590149, 0.9990184903144836, 0.9814757108688354, 0.9951430559158325, 0.9974409937858582, 0.9987649321556091, 0.9921976327896118, 0.9994720220565796, 0.9829403162002563, -0.19648300111293793, 0.9990527629852295, 0.9982097744941711, 0.9990559816360474, 0.9994577169418335, 0.9995303153991699, 0.999024510383606, 0.999701201915741, 0.9995482563972473, 0.9994628429412842, 0.9992148280143738, 0.9995391964912415, 0.9977765679359436, 0.998933732509613, 0.9977701306343079, 0.9987254738807678, 0.9986830949783325, 0.9995498657226562, 0.9993309378623962, 0.98836350440979, 0.9991179704666138, 0.9991480708122253, 0.9988345503807068, 0.9990955591201782, 0.9999071359634399, 0.9994404315948486, 0.999164342880249, 0.9983203411102295, 0.9991223812103271, 0.9369516372680664, 0.9996386766433716]
y28 = [0.9857504367828369, 0.9943026304244995, 0.9923352599143982, 0.999471127986908, 0.9598574042320251, 0.7935185432434082, 0.9991947412490845, 0.3265247046947479, 0.9986054301261902, 0.9747564792633057, 0.9848558306694031, 0.8383654356002808, 0.9977190494537354, 0.991989016532898, 0.9592092633247375, 0.9986923336982727, 0.9965144991874695, 0.9937230944633484, 0.9963220357894897, 0.9469496607780457, 0.9854831099510193, 0.9791462421417236, 0.9894205331802368, 0.9435108304023743, 0.9862751960754395, 0.994020402431488, 0.9965869188308716, 0.9838359951972961, 0.9995354413986206, 0.9824126958847046, 0.9551329016685486, 0.982964038848877, 0.9993656873703003, 0.9958580732345581, 0.9957500696182251, 0.9754493236541748, 0.9786953926086426, 0.9997802376747131]
y29 = [0.9993963241577148, 0.9989535808563232]
y30 = [0.9915195107460022, 0.9891536235809326, 0.9900689721107483]
y31 = [0.9768092632293701, 0.999238908290863, 0.9990508556365967, 0.9625977873802185, 0.9988071918487549]
y32 = [0.9985264539718628, 0.9972355365753174, 0.9997188448905945, 0.9979760050773621, 0.9984445571899414, 0.9961658716201782, 0.9993590712547302]
y33 = [0.9977817535400391, 0.9952450394630432, 0.9984519481658936]
y34 = [0.982841968536377, 0.6315540075302124, 0.9888501763343811, 0.9981498718261719, 0.9963344931602478, 0.020768536254763603, 0.9976814389228821, 0.9506384134292603, 0.9914975762367249, 0.9997918605804443]
y35 = [0.9925666451454163, 0.9961777925491333, 0.9870061874389648, 0.9957901239395142, 0.9690848588943481, 0.9941044449806213, 0.9859509468078613, 0.9772804975509644, 0.974338173866272, 0.9882317781448364, 0.9848266243934631, 0.9924362897872925, 0.9930825233459473, 0.9956797957420349, 0.9993630647659302, 0.9885126352310181, 0.9899575710296631, 0.91970294713974, 0.9976682066917419, 0.9733282327651978, 0.9396858215332031, 0.943569540977478, 0.9761271476745605, 0.9873608946800232, 0.9927992820739746, 0.9953224658966064, 0.9302277565002441, 0.9965407252311707, 0.9703083038330078, 0.9909937977790833, 0.9643782377243042]
y36 = [0.936760425567627, 0.987034797668457, 0.9907920360565186, 0.7052257657051086, 0.9890735745429993, 0.9908177256584167, 0.9860473275184631, 0.9929934144020081, 0.9858899116516113, 0.8168829679489136, 0.980478048324585, 0.9923626184463501, 0.9979971051216125, 0.9992071390151978, 0.990861177444458, 0.9778026938438416, 0.9998721480369568, 0.9895764589309692, 0.9991170167922974, 0.9857484698295593, 0.9860245585441589, 0.9945433735847473, 0.9989933371543884, 0.9895479083061218, 0.9985276460647583, 0.993880033493042, 0.9919582009315491, 0.9967314600944519, 0.9963304996490479, 0.993938684463501, 0.9173802137374878, 0.5346832275390625, 0.9945269227027893, 0.4739757478237152, 0.9979779720306396, 0.9894078969955444, 0.9561435580253601, 0.9962385892868042, 0.9902626276016235, 0.985448956489563, 0.9514222741127014, 0.9190465807914734, 0.9984123706817627, 0.999413788318634, 0.9890195727348328, 0.9841238856315613, 0.9955131411552429, 0.9955273866653442, 0.9966627955436707]
y37 = [0.989202082157135, 0.9965443015098572, 0.9998112916946411, 0.9999334812164307, 0.9997839331626892, 0.9978452920913696, 0.9989116787910461, 0.9988851547241211]
y38 = [0.9910422563552856, 0.9961806535720825, 0.9969102144241333, 0.9872214794158936, 0.9913026094436646, 0.9529107213020325, 0.9957379102706909, 0.9947993755340576, 0.9861795902252197, 0.9340552091598511, 0.9921315312385559, 0.9991650581359863, 0.9982781410217285, 0.9667629599571228, 0.9792617559432983, 0.9971988201141357, 0.9927997589111328, 0.9949190616607666, 0.9979551434516907, 0.9268889427185059, 0.9945069551467896]
y39 = [0.9989058971405029, 0.9897583723068237, 0.9643458127975464, 0.9823453426361084, 0.99058598279953, 0.9403995275497437, 0.9845662117004395, 0.9294034838676453, 0.8976594805717468, 0.9135034680366516, 0.9805866479873657, 0.9977810978889465, 0.9968785047531128, 0.9921783804893494, 0.20876789093017578, 0.9728947877883911, 0.9993656873703003, 0.9669915437698364, 0.9682530760765076, 0.9940041303634644, 0.9994118809700012, 0.9965462684631348, 0.9942935705184937, 0.5374487042427063, 0.9953093528747559, 0.9208367466926575, 0.9976258277893066, 0.9990164041519165, 0.9925106167793274, 0.9923762679100037, 0.9780343174934387, 0.9764603972434998, 0.9787130951881409, 0.9806484580039978, 0.9908338785171509, 0.9974124431610107, 0.9899915456771851, 0.9968061447143555, 0.9955919981002808, 0.9875558614730835, 0.9940754175186157, 0.9992934465408325, 0.9959099888801575, 0.9957774877548218, 0.965304434299469, 0.997353732585907, 0.988699734210968, 0.9879360795021057, 0.9983607530593872, 0.9951983690261841, 0.989128053188324, 0.8569532632827759, 0.9949854016304016, 0.9948146343231201, 0.9962407946586609, 0.9965056777000427, 0.9821833968162537, 0.9818890690803528, 0.9991148114204407, 0.953906774520874, 0.9922763705253601, 0.9994741678237915, 0.9952937364578247, 0.97474205493927, 0.975516676902771, 0.8867279291152954, 0.9895895719528198, 0.9989984631538391, 0.9953665733337402, 0.954923689365387, 0.9978188276290894, 0.9990936517715454, 0.9987694621086121, 0.9989590048789978, 0.9995070099830627, 0.9675737619400024, 0.9955877661705017, 0.9966704249382019, 0.6081538796424866, 0.9988265037536621]


s1 = pd.Series(np.array(y1))
s2 = pd.Series(np.array(y2))
s3 = pd.Series(np.array(y3))
s4 = pd.Series(np.array(y4))
s5 = pd.Series(np.array(y5))
s6 = pd.Series(np.array(y6))
s7 = pd.Series(np.array(y7))
s8 = pd.Series(np.array(y8))
s9 = pd.Series(np.array(y9))
s10 = pd.Series(np.array(y10))
s11 = pd.Series(np.array(y11))
s12 = pd.Series(np.array(y12))
s13 = pd.Series(np.array(y13))
s14 = pd.Series(np.array(y14))
s15 = pd.Series(np.array(y15))
s16 = pd.Series(np.array(y16))
s17 = pd.Series(np.array(y17))
s18 = pd.Series(np.array(y18))
s19 = pd.Series(np.array(y19))
s20 = pd.Series(np.array(y20))
s21 = pd.Series(np.array(y21))
s22 = pd.Series(np.array(y22))
s23 = pd.Series(np.array(y23))
s24 = pd.Series(np.array(y24))
s25 = pd.Series(np.array(y25))
s26 = pd.Series(np.array(y26))
s27 = pd.Series(np.array(y27))
s28 = pd.Series(np.array(y28))
s29 = pd.Series(np.array(y29))
s30 = pd.Series(np.array(y30))
s31 = pd.Series(np.array(y31))
s32 = pd.Series(np.array(y32))
s33 = pd.Series(np.array(y33))
s34 = pd.Series(np.array(y34))
s35 = pd.Series(np.array(y35))
s36 = pd.Series(np.array(y36))
s37 = pd.Series(np.array(y37))
s38 = pd.Series(np.array(y38))
s39 = pd.Series(np.array(y39))




# combinedata = [y1, y2]
# group = [zeros(size(y1)) ones(size(y2))]
# boxplot(combinedata,group)
data = pd.DataFrame({"DCCA": s1, "DCCR": s2, "DMAA": s3, "DRAC": s4, "DRVA": s5, "DRWV": s6, "HBRN": s7, "HCOM": s8, "HDIM": s9, "HDMS": s10, "HEXP": s11, "HIMS": s12, "HOTH": s13, "OAAN": s14, "OAID": s15, "OAIS": s16, "OEDE": s17, "OFFN": s18, "OFPF": s19, "OFPO": s20, "OICD": s21, "OILN": s22, "OIRO": s23, "OITC": s24, "OLLN": s25, "OMOP": s26, "ORRN": s27, "SDFN": s28, "SDIB": s29, "SDIF": s30, "SDLA": s31, "SIIF": s32, "SIRT": s33, "SISA": s34, "SISF": s35, "SMOV": s36, "SMVB": s37, "SRIF": s38, "STYP": s39})
data.boxplot()
plt.xlabel('x')
plt.ylabel('y')
plt.grid(linestyle="--", alpha=0.3)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel("Defect Types",fontsize=10)
plt.ylabel("Similarity",fontsize=10)
plt.show()
df = pd.DataFrame(data)
pd.set_option('display.max_columns', None)
print(df.describe().round(3))