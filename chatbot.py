#!/usr/bin/env python3

from nltk.tokenize import word_tokenize
from stem import stem
from manim.utils.color import Colors,random_bright_color
from manim import *
from manim_fonts import *
import random

class Combined(Scene):
    def construct(self):
        Chat.construct(self)
        Overview.construct(self)
        NLPTheory.construct(self)
        BagOfWords.construct(self)
        Intent.construct(self)
        Entity.construct(self)
        TrainingData.construct(self)
        Model.construct(self)
        Demo.construct(self)

def message(color,text_color, string):
    with RegisterFont("Quicksand") as fonts:
        result = VGroup() # create a VGroup
        text = Text(string,color=text_color,font=fonts[0])
        rect = SurroundingRectangle(text, corner_radius=0.3,color=color,buff=MED_SMALL_BUFF)
        result.add(text,rect)
        return result

def comp(text):
    rect = Rectangle(width=2,height=1)
    text = Paragraph(*text.split(" "),alignment="center").scale(0.8)
    text.move_to(rect.get_center())
    return VGroup(rect,text)

def vg_user(text):
    c = Circle(radius=0.5)
    user_t = Text(text).next_to(c,DOWN)
    return VGroup(c,user_t)

def lower(lst):
    return [s.lower() for s in lst]

def depunk(lst):
    res = []
    p = [".",",",":","?","!","\"","\'"]
    for s in lst:
        if s not in p:
            res.append(s)
    return res

def stemming(lst):
    return [stem(s) for s in lst]

def pipeline_fn(sentence):
    fn_pipeline = [stemming,lower,depunk]
    lst = word_tokenize(sentence)
    for fn in fn_pipeline:
        lst = fn(lst)
    return lst

def bag_of_words(lst,words):
    bow = dict.fromkeys(words,0)
    for word in lst:
        bow[word]=words.count(word)
    return bow

def comp_l(lst):
    text = Paragraph(*lst).scale(0.8)
    rect = SurroundingRectangle(text,buff=0.3,color=WHITE)
    return VGroup(rect,text)

def intents(tag,patterns,responses,title=False):
    tag_c = comp(tag)
    patterns_c = comp_l(patterns)
    responses_c = comp_l(responses)

    return VGroup(tag_c,patterns_c,responses_c)

def patterns2wordset(patterns):
    lst = []
    for pattern in patterns:
        lst = lst+pipeline_fn(pattern)
    return list(set(lst))

def node():
    return Circle(radius=0.2)

def label(text,obj):
    text_o = Text(text).next_to(obj,2*DOWN)
    return VGroup(SurroundingRectangle(VGroup(text_o,obj)),text_o)

def neural_net(input_size,hidden_size,output_size,n_hidden,buff_h,buff_v):
    inputs = [node() for i in range(input_size)]
    hidden_layers = []
    for i in range(n_hidden):
        hidden_layers.append([node() for j in range(hidden_size)])

    outputs = [node() for i in range(output_size)]

    inputs_vg = VGroup(*inputs).arrange(DOWN,buff=buff_h)

    hidden_layers_vgs = []
    for hidden_layer in hidden_layers:
        hidden_layers_vgs.append(VGroup(*hidden_layer).arrange(DOWN,buff=buff_h))
    hidden_vg = VGroup(*hidden_layers_vgs).arrange(RIGHT,buff=buff_v)
    print(hidden_vg)

    outputs_vg = VGroup(*outputs).arrange(DOWN,buff=buff_h)



    lines_in = []
    lines_h = []
    lines_o = []
    net_vg = VGroup(inputs_vg,hidden_vg,outputs_vg).arrange(RIGHT,buff=buff_v)

    for inp in inputs_vg:
        for h in hidden_vg[0]:
            lines_in.append(Line(inp,h,stroke_width=0.5))

    for idx,h_l in enumerate(hidden_vg[:-1]):
        h_n = hidden_vg[idx+1]
        for h in h_l:
            for hn in h_n:
                lines_h.append(Line(h,hn,stroke_width=0.5))

    for h in hidden_vg[-1]:
        for out in outputs_vg:
            lines_o.append(Line(h,out,stroke_width=0.5))

    #print(lines_in_h)
    lines_ivg = VGroup(*lines_in)
    lines_hvg = VGroup(*lines_h)
    lines_ovg = VGroup(*lines_o)

    net_vg.add(lines_ivg,lines_hvg,lines_ovg)

    return net_vg

class Demo(Scene):

    def construct(self):
        bg_color = "#001514"
        self.camera.background_color = bg_color
        Text.set_default(font_size=30)
        init = Rectangle(width=0).move_to(4.5*UP+7.5*RIGHT)

        _,dlg = Overview.dialog(self,1,init,"Teraz si uk????eme ako vyu??i?? tento model")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))

        input_size=7
        hidden_size=4
        output_size=5
        n_hidden=1
        buff_h=0.3
        buff_v=1
        model = neural_net(input_size,hidden_size,output_size,n_hidden,buff_h,buff_v)
        self.play(Write(model))



        _,dlg = Overview.dialog(self,1,init,"ako m????eme pomocou natr??novan??ho modelu neuronovej siete vytvori?? Chatbot")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))

        softmax = Rectangle(width=1,height=4).next_to(model,2*RIGHT)
        softmax_t = Text("Softmax").rotate(PI/2).move_to(softmax.get_center())
        model.add(softmax,softmax_t)
        self.play(Write(softmax),Write(softmax_t))

        model_abs = comp("Model")
        self.play(Transform(model,model_abs))

        _,dlg = Overview.dialog(self,1,init,"Pre ka??d?? vstup od u??ivate??a sa veta transformuje na Bag of Words vektor")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))

        input_text = Text("Ahoj Chatbot").next_to(model,3*LEFT)
        self.play(Write(input_text))
        all_words = patterns2wordset(["Ahoj.", "Ako sa m????.", "Dobr?? de??."])
        transformed = pipeline_fn("Ahoj Chatbot")
        bow = bag_of_words(transformed,all_words)


        self.play(model.animate.shift(RIGHT))

        bow_t = Text(f"{bow}").scale(0.7).move_to(input_text.get_center()+LEFT)
        bow_v = Text(f"{list(bow.values())}").scale(0.7).move_to(input_text.get_center()+LEFT)
        self.play(Transform(input_text,bow_t))
        self.play(Transform(input_text,bow_v))

        _,dlg = Overview.dialog(self,1,init,"Tento vektor vie model ????ta?? do vstupu")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))

        self.play(Transform(input_text, model))
        output = [0.94,0.03,0.2,0.1]
        output_v = VGroup(*[Text(str(o)) for o in output]).arrange(DOWN,buff=0.4).next_to(model,4*RIGHT)
        self.play(Transform(model.copy(),output_v))


        _,dlg = Overview.dialog(self,1,init,"V??stupn?? hodnoty s?? predpove?? modelu a s?? to pravdepodobnosti pre ka??d?? intent")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))

        main = comp("Intent").next_to(model,5*LEFT+UP).scale(0.7)
        attrs_example = intents("Zdravenie",["Ahoj.", "Ako sa m????.", "Dobr?? de??."],["Ahoj." ,"Ako V??m m????em pom??c???" ,"Ahoj, ako V??m m????em pom??c???"]).scale(0.7)
        attrs_example.arrange(2*RIGHT).next_to(main,3*DOWN)


        lines_example = VGroup(*[DashedLine(main.get_bottom(),attr.get_top()) for attr in attrs_example])

        self.play(Write(main),Write(attrs_example),Write(lines_example))

        _,dlg = Overview.dialog(self,1,init,"Z tochto vektora si vyberieme najv??????iu hodnotu a kore??ponduj??ci intent")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))

        selected = SurroundingRectangle(attrs_example[0])
        self.play(Write(selected))

        _,dlg = Overview.dialog(self,1,init,"Ke?? je u?? intent vybran??, pozrieme sa na zoznam mo??n??ch odpoved??")
        self.wait()

        selected_response = SurroundingRectangle(attrs_example[2])
        self.play(Transform(selected,selected_response))
        self.play(FadeOut(dlg, shift=UP))

        _,dlg = Overview.dialog(self,1,init,"Z tohto zoznamu si m????eme n??hodne vybra?? jednu odpove??")
        self.wait()


        selected_rand = SurroundingRectangle(random.choice(attrs_example[2]))
        self.play(FadeOut(dlg, shift=UP))
        self.play(Transform(selected,selected_rand))

        _,dlg = Overview.dialog(self,1,init,"a n??sledne ju vyp??sa?? ako odpove?? na v??stup.")
        self.wait()



        self.play(Transform(selected,output))
        self.play(FadeOut(dlg, shift=UP))

 class Model(Scene):

    def construct(self):
        bg_color = "#001514"
        self.camera.background_color = bg_color
        Text.set_default(font_size=30)
        init = Rectangle(width=0).move_to(4.5*UP+6.5*RIGHT)
        
        _,dlg = Overview.dialog(self,1,init,"Teraz sme u?? pripraven?? vytvori?? model ktor?? budeme tr??nova??")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))

        _,dlg = Overview.dialog(self,1,init,"Pou??ijeme jednoduch?? dopredn?? neuonov?? sie??")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))

        input_size=7
        hidden_size=4
        output_size=5
        n_hidden=1
        buff_h=0.3
        buff_v=1
        model = neural_net(input_size,hidden_size,output_size,n_hidden,buff_h,buff_v)
        self.play(Write(model))

        softmax = Rectangle(width=1,height=4).next_to(model,2*RIGHT)
        self.play(Write(softmax))


        all_words = patterns2wordset(["Ahoj.", "Ako sa m????.", "Dobr?? de??."])
        transformed = pipeline_fn("Dobr?? de??.")
        bow = bag_of_words(transformed,all_words)
        bow_t = Text(f"{list(bow.values())}").next_to(model,2*LEFT)
        output_vec = [0.80,0.08,0.06,0.04,0.02]
        output_texts = [Text(str(val)) for val in output_vec]
        output = VGroup(*output_texts).arrange(DOWN,buff=0.5).next_to(softmax,2*RIGHT)



        self.play(Write(bow_t))


        l1 = label("Vstupn?? vektor",bow_t)
        l2 = label("Po??et v??etk??ch slov",model[0])
        l3 = label("Skryt?? vrstva",model[1])
        l4 = label("Po??et intentov",model[2])
        l5 = label("Softmax",softmax)
        l6 = label("V??stupn?? vektor",output)

        self.play(Write(l1))
        self.play(ReplacementTransform(l1,l2))
        self.play(ReplacementTransform(l2,l3))
        self.play(ReplacementTransform(l3,l4))
        self.play(ReplacementTransform(l4,l5))

        _,dlg = Overview.dialog(self,1,init,"Posielame vstup do siete.")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))

        self.play(ReplacementTransform(bow_t,model[0]))
        li = model[3].copy().set_stroke(color=YELLOW, width=3)
        self.play(ShowPassingFlash(li))
        hi = model[4].copy().set_stroke(color=YELLOW, width=3)
        self.play(ShowPassingFlash(hi))
        oi = model[5].copy().set_stroke(color=YELLOW, width=3)
        self.play(ShowPassingFlash(oi))

        self.play(ReplacementTransform(model[2].copy(),softmax))
        self.play(ReplacementTransform(softmax.copy(),output))

        self.play(ReplacementTransform(l5,l6))
        self.wait()
        self.play(*[FadeOut(mob)for mob in self.mobjects])

class TrainingData(Scene):

    def construct(self):
        bg_color = "#001514"
        self.camera.background_color = bg_color
        Text.set_default(font_size=30)
        init = Rectangle(width=0).move_to(4.5*UP+6.5*RIGHT)

        _,dlg = Overview.dialog(self,1,init,"Aby sme mohli vytvori?? model pre Chatbota, mus??me definova?? tr??novacie d??ta")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))

        _,dlg = Overview.dialog(self,1,init,"Definujme X ako vstup a Y ako v??stup pre model")
        self.wait(2)
        self.play(FadeOut(dlg, shift=UP))

        x = Text("X").scale(2)
        y = Text("Y").scale(2)


        io_vg = VGroup(x,y).arrange(RIGHT,buff=6)

        self.play(Write(io_vg))

        self.play(io_vg.animate.shift(3*DOWN))

        _,dlg = Overview.dialog(self,1,init,"Vstupom by bola veta a v??stupom preddefinovan?? intent")
        self.wait(2)
        self.play(FadeOut(dlg, shift=UP))

        veta_x = Text("Dobr?? de??.").next_to(x,10*UP)
        intent_y = Text("Zdravenie").next_to(y,10*UP)
        arrow_veta = Arrow(veta_x.get_right(),intent_y.get_left(),buff=1.5)
        arrow_veta.add_updater(
            lambda mobject: mobject.put_start_and_end_on(veta_x.get_right()+RIGHT,intent_y.get_left()+LEFT)
        )
        veta_vg = VGroup(veta_x,arrow_veta,intent_y)

        self.play(Write(veta_vg))

        _,dlg = Overview.dialog(self,1,init,"Vety transformujeme na vektor Bag of Words pomocou intentu Zdravenie")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))

        #attrs_example = intents("Zdravenie",["Ahoj.", "Ako sa m????.", "Dobr?? de??."],["Ahoj." ,"Ako V??m m????em pom??c???" ,"Ahoj, ako V??m m????em pom??c???"]).next_to(dlg,DOWN)
        #self.play(Write(attrs_example))

        all_words = patterns2wordset(["Ahoj.", "Ako sa m????.", "Dobr?? de??."])
        transformed = pipeline_fn("Dobr?? de??.")
        bow = bag_of_words(transformed,all_words)
        bow_t = Text(f"{bow}").move_to(veta_x.get_center())
        self.play(Transform(veta_x,bow_t))
        self.wait()
        bow_t_v = Text(f"{list(bow.values())}").move_to(veta_x.get_center())
        self.play(Transform(veta_x,bow_t_v))
        self.wait()

        title_x = Text("Vektor Bag of Words").next_to(veta_x,3*UP)
        title_y = Text("Intent").next_to(intent_y,3*UP)

        self.play(FadeIn(title_x),FadeIn(title_y))
        self.play(*[FadeOut(mob)for mob in self.mobjects])

class Entity(Scene):

    def construct(self):
        bg_color = "#001514"
        self.camera.background_color = bg_color
        Text.set_default(font_size=30)
        init = Rectangle(width=0).move_to(4.5*UP+5.5*RIGHT)

        _,dlg = Overview.dialog(self,1,init,"N??sleduje Entita")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))

        entity = comp("Entita")
        self.play(Write(entity))

        _,dlg = Overview.dialog(self,1,init,"Podobne ako intent, Entitu si mi ur??ujeme a po??et Ent??t je ??ubovo??n??")
        self.wait(2)
        self.play(FadeOut(dlg, shift=UP))

        _,dlg = Overview.dialog(self,1,init,"Ako pr??klad si uvedieme tri entity")
        self.wait(2)
        self.play(FadeOut(dlg, shift=UP))

        entities_labels = ["OSOBA", "ORGANIZ??CIA","LOKALITA", "??AS"]
        entities = []
        for label in entities_labels:
            entities.append(comp(label))
        entities_vg = VGroup(*entities).arrange(RIGHT,buff=0.4)
        self.play(ReplacementTransform(entity,entities_vg))

        _,dlg = Overview.dialog(self,1,init,"Na rozdiel od entite, kde jedna veta m?? jeden intent")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))

        _,dlg = Overview.dialog(self,1,init,"po??et Ent??t vo vete m????e by?? viacero")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))

        _,dlg = Overview.dialog(self,1,init,"Dajme si znovu vetu na pr??klad")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))


        #veta = "Chcel by som si objedna?? jeden l??stok pre Michala na vlak do Bratislavy na zajtra r??no."
        #text = Text(veta).next_to(entities_vg,3*DOWN)
        #text = VGroup(*[Text(part) for part in veta]).arrange_in_grid(1,len(veta),row_alignments="c").next_to(entities_vg,3*DOWN)
        #text =Text(veta).next_to(entities_vg,3*DOWN)
        veta = ["Chcel by som si objedna?? jeden l??stok pre ", "Michala", "na vlak do", "Bratislavy", "na", "zajtra r??no."]
        text = VGroup(*[Text(part) for part in veta]).arrange_in_grid(1,len(veta),row_alignments="c").next_to(entities_vg,3*DOWN)
        self.play(Write(text))

        init.shift(RIGHT)
        _,dlg = Overview.dialog(self,1,init,"Pre analyzovanie vety sa pou????va model pre rozpozn??vanie pomenovanej entity")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))
        ent_idx = [1,3,5]
        #veta = ["Chcel by som si objedna?? jeden l??stok pre ", "Michala", "na vlak do", "Bratislavy", "na", "zajtra r??no."]
        #text_a = VGroup(*[Text(part) for part in veta]).arrange_in_grid(1,len(veta),row_alignments="c").move_to(text.get_center())
        #self.play(Transform(text,text_a))
        for i in ent_idx:
            self.play(Write(SurroundingRectangle(text[i])))

        self.play(entities_vg[0].animate.next_to(text[1],DOWN),entities_vg[2].animate.next_to(text[3],DOWN),entities_vg[3].animate.next_to(text[5],DOWN))
        self.wait()

        _,dlg = Overview.dialog(self,1,init,"Pre dan?? vetu sme rozpoznali tri Entity")
        self.wait()
        self.play(FadeOut(dlg, shift=UP))
        self.play(*[FadeOut(mob)for mob in self.mobjects])

class Intent(Scene):

    def construct(self):
        bg_color = "#001514"
        self.camera.background_color = bg_color
        Text.set_default(font_size=30)
        init = Rectangle(width=0).move_to(4.5*UP+5.5*RIGHT)

        _,dlg = Overview.dialog(self,1,init,"Teraz si uk????eme ??o je Intent")
        self.play(FadeOut(dlg, shift=UP))
        self.wait()

        main = comp("Intent")
        self.play(Write(main))

        _,dlg = Overview.dialog(self,1,init,"Intent sa pou????va, aby ChatBot vedel ??o je z??mer rozhovoru.")
        self.play(FadeOut(dlg, shift=UP))
        self.wait()

        _,dlg = Overview.dialog(self,1,init,"M????eme ma?? ??ubovo??n?? po??et z??merov.")

        self.wait()

        main.save_state()
        main_multiple = VGroup(*[comp("Intent") for i in range(5)])
        main_multiple.add(Text("...").scale(2))
        main_multiple.arrange(RIGHT)
        self.play(Transform(main,main_multiple))
        self.wait()
        self.play(Restore(main))
        self.play(FadeOut(dlg, shift=UP))

        _,dlg = Overview.dialog(self,1,init,"Ka??d?? intent m?? svoj tag, vzory a odpovede.")
        self.wait()

        attrs = intents("Tag",["Vzory"],["Odpovede"])
        attrs.arrange(2*RIGHT).next_to(main,3*DOWN)
        lines = VGroup(*[DashedLine(main.get_bottom(),attr.get_top()) for attr in attrs])
        self.play(Write(attrs),Write(lines))

        self.play(FadeOut(dlg, shift=UP))
        _,dlg = Overview.dialog(self,1,init,"Dajme si pr??klad jedn??ho intentu.")
        self.wait()

        attrs_example = intents("Zdravenie",["Ahoj.", "Ako sa m????.", "Dobr?? de??."],["Ahoj." ,"Ako V??m m????em pom??c???" ,"Ahoj, ako V??m m????em pom??c???"])
        attrs_example.arrange(2*RIGHT).next_to(main,3*DOWN)
        tag_t = Text("Tag").next_to(attrs_example[0],DOWN)
        vzory_t = Text("Vzory").next_to(attrs_example[1],DOWN)
        odp_t = Text("Odpovede").next_to(attrs_example[2],DOWN)
        titles = VGroup(tag_t,vzory_t,odp_t)


        lines_example = VGroup(*[DashedLine(main.get_bottom(),attr.get_top()) for attr in attrs_example])
        self.play(Transform(attrs,attrs_example),Transform(lines,lines_example))
        self.play(FadeIn(titles))
        self.wait()

        self.play(FadeOut(dlg, shift=UP))
        _,dlg = Overview.dialog(self,1,init,"Ka??d?? intent m?? svoj tag, vzory a odpovede.")
        self.wait()
        self.play(*[FadeOut(mob)for mob in self.mobjects])



class BagOfWords(Scene):


    def construct(self):
        bg_color = "#001514"
        self.camera.background_color = bg_color
        Text.set_default(font_size=30)
        init = Rectangle(width=0).move_to(4.5*UP+5.5*RIGHT)
        _,dlg = Overview.dialog(self,1,init,"Nasledovn?? pojem je Bag of Words")
        self.play(FadeOut(dlg, shift=UP))
        self.wait()

        _,dlg = Overview.dialog(self,1,init,"Po vytvoren?? po??a re??azcov, vytvor??me vektor Bag of Words nasledovne.")
        self.play(FadeOut(dlg, shift=UP))
        self.wait()

        fn_pipeline = [word_tokenize,stemming,lower,depunk]
        veta1 = "Prv?? kr??sna veta, ktor?? pou??ijeme."
        veta2 = "Druh?? e??te kraj??ia veta, ktor?? pou??ijeme."
        veta1_m = Text(veta1)
        veta2_m = Text(veta2)

        vety = VGroup(veta1_m,veta2_m).arrange(DOWN)
        self.play(Write(vety))
        veta1 = pipeline_fn(veta1)
        veta2 = pipeline_fn(veta2)

        veta1_m = Text(f"{veta1}")
        veta2_m = Text(f"{veta2}")

        vety_t = VGroup(veta1_m,veta2_m).arrange(DOWN).next_to(vety,3*DOWN)
        vety_t.add_updater(
            lambda mobject: mobject.next_to(vety,3*DOWN)
        )
        self.play(ReplacementTransform(vety.copy(),vety_t))
        self.wait()

        _,dlg = Overview.dialog(self,1,init,"Dan?? polia zoskup??me bez duplicitn??ch slov")
        self.play(FadeOut(dlg, shift=UP))
        self.wait()

        set_w = list(set(veta1+veta2))
        vety_set = Text(f"{set_w}")
        #self.play(ReplacementTransform(vety_t,vety_set))
        #self.wait()
        self.play(vety.animate.shift(3*LEFT+UP))
        bag = Rectangle(width=4.0,height=5.5).move_to(4.5*RIGHT)
        self.play(Write(bag))

        #words = VGroup(*[Text(w) for w in set_w]).arrange(DOWN).move_to(bag.get_center())
        words = Paragraph(*set_w,line_spacing=1.5).move_to(bag.get_center()+LEFT)
        counts = Paragraph(*set_w,line_spacing=1.5).move_to(bag.get_center()+LEFT)
        self.play(ReplacementTransform(vety_t.copy(),words))
        self.wait()

        bow = bag_of_words(set_w,veta1+veta2)
        print(veta1+veta2)
        print(bow)
        bow_p = Paragraph(*map(str,bow.values()),line_spacing=1.5).move_to(bag.get_center()+RIGHT)

        init.shift(3*LEFT)
        _,dlg = Overview.dialog(self,1,init,"N??sledne spo????tame po??et v??skytov slov pre obe vety.")


        self.play(ReplacementTransform(vety_t.copy(),bow_p))
        self.wait()

        self.play(FadeOut(dlg, shift=UP))
        self.wait()
        _,dlg = Overview.dialog(self,1,init,"T??mto sme vytvorili vektor Bag of Words")

        vector_bow = Text(f"{list(bow.values())}").next_to(vety_t,3*DOWN)
        self.play(ReplacementTransform(bow_p.copy(),vector_bow))
        self.wait()
        self.play(*[FadeOut(mob)for mob in self.mobjects])

class NLPTheory(Scene):


    def construct(self):

        bg_color = "#001514"
        self.camera.background_color = bg_color
        Text.set_default(font_size=30)
        init = Rectangle(width=0).move_to(4.5*UP+5.5*RIGHT)
        _,dlg = Overview.dialog(self,1,init,"Najprv tokeniz??cia")
        self.play(FadeOut(dlg, shift=UP))
        self.wait()
        _,dlg = Overview.dialog(self,1,init,"Je to rozdelenie re??azca na zmyslupln?? ??asti.")

        self.wait()

        veta = "Jeden pr??klad vety, ktor?? pou????va viacero slov."

        sentence = Text(veta).scale(0.8)
        tokenizer = comp("Tokenizing")
        stemmer = comp("Stemming")

        group = VGroup(sentence,tokenizer).arrange(DOWN,buff=0.5).move_to(UP)
        self.play(Write(group))

        tokenized_s = word_tokenize(veta)
        tokenized = ["".join(["\'", word]) for word in tokenized_s]
        tokenized = "\', ".join(tokenized)
        tokenized = f"[{tokenized}\']"
        random_colors = []
        for s in tokenized_s:
            r_color = random_color()
            while r_color == bg_color:
                r_color = random_color()
            random_colors.append({s:r_color})
        d = {}
        for color in random_colors:
            d.update(color)
        sentence_t = Text(tokenized,t2c=d).scale(0.7).next_to(tokenizer,1.5*DOWN)

        self.play(ReplacementTransform(sentence.copy(),tokenizer[0]))
        self.play(ReplacementTransform(tokenizer[0].copy(),sentence_t))
        self.wait()

        stemmer.next_to(sentence_t,1.5*DOWN)

        self.play(FadeOut(dlg, shift=UP))
        _,dlg = Overview.dialog(self,1,init,"Stemming je vytvorenie kore??a slova. Nech??va sa iba hlavn?? ??as?? slova.")

        stemmed_l = [stem(slovo) for slovo in tokenized_s]

        stemmed = ["".join(["", word]) for word in stemmed_l]
        stemmed = f"{stemmed}"
        sd = {}
        for idx,token in enumerate(tokenized_s):
            sd[stemmed_l[idx]] = d[token]
        stemmed = Text(stemmed,t2c=sd).scale(0.7).next_to(stemmer,1.5*DOWN)
        self.play(ReplacementTransform(sentence_t.copy(),stemmer[0]))
        self.play(ReplacementTransform(stemmer[0].copy(),stemmed))
        self.wait()
        self.play(FadeOut(stemmed),FadeOut(sentence_t))

        nlp_comps = ["Tokenize","Stemming","lowercase","depunktuovanie"]
        nlp_pipeline = VGroup(*[comp(c) for c in nlp_comps]).arrange(DOWN,buff=0.5).move_to(4*RIGHT+0.5*DOWN)
        self.play(Transform(VGroup(tokenizer,stemmer),nlp_pipeline))
        self.play(sentence.animate.shift(2*LEFT,2*DOWN).scale(1.1))
        self.wait()
        self.play(FadeOut(dlg, shift=UP))
        _,dlg = Overview.dialog(self,1,init,"Cel?? proces NLP procesovania vyzer?? nasledovne.")

        arrows = []
        for idx,c in enumerate(nlp_pipeline):
            if idx < len(nlp_pipeline)-1:
                c_arrow = CurvedArrow(c.get_right(),nlp_pipeline[idx+1].get_right(),radius = -1)
                arrows.append(c_arrow)
            arrows_g = VGroup(*arrows)
        self.play(Write(arrows_g))
        self.play(ReplacementTransform(sentence.copy(),nlp_pipeline[0]))
        self.play(nlp_pipeline[0][0].animate.set_color(YELLOW))
        self.play(nlp_pipeline[0][0].animate.set_color(WHITE))

        fn_pipeline = [word_tokenize,stemming,lower,depunk]
        veta = fn_pipeline[0](veta)
        n_sentence = Text(f"{veta}").scale(0.8).move_to(sentence.get_center())
        self.play(ReplacementTransform(sentence,n_sentence))
        sentence = n_sentence
        for idx,arrow in enumerate(arrows_g):
            nlp_c = nlp_pipeline[idx+1][0]
            self.play(arrow.animate.set_color(YELLOW))
            self.play(nlp_c.animate.set_color(YELLOW))
            self.play(arrow.animate.set_color(WHITE))
            self.play(nlp_c.animate.set_color(WHITE))
            veta = fn_pipeline[idx+1](veta)
            n_sentence = Text(f"{veta}").scale(0.8).move_to(sentence.get_center())
            self.play(ReplacementTransform(sentence,n_sentence))
            sentence = n_sentence
        self.wait()
        self.play(FadeOut(dlg, shift=UP))
        self.play(*[FadeOut(mob)for mob in self.mobjects])

class Overview(Scene):

    def dialog(self, person, last, string):
        Text.set_default(font_size=24)
        dots = Text("...").scale(1.5)
        if person == 1:
            dialog = message(Colors.teal_e.value,WHITE,string).next_to(last,DOWN+2*LEFT)
            dots.move_to(dialog.get_center())
            self.play(FadeIn(dots))
            self.play(Unwrite(dots))
            self.play(FadeIn(dialog, shift=UP))
            self.wait()
            #last = last.move_to(last.get_center()+DOWN)
            return last,dialog
        else:
            dialog = message(Colors.gray.value,WHITE,string).next_to(last,DOWN+2*RIGHT)
            dots.move_to(dialog.get_center())
            self.play(FadeIn(dots))
            self.play(FadeOut(dots))
            self.play(FadeIn(dialog, shift=UP))
            self.wait()
            #last = last.move_to(last.get_center()+DOWN)
            return last,dialog


    def construct(self):
        bg_color = "#001514"
        self.camera.background_color = bg_color
        Text.set_default(font_size=25)
        init = Rectangle(width=2).move_to(4.5*UP)
        last,dialog = Overview.dialog(self,2, init, "Naprv si uk????eme cel?? obraz")



        user = vg_user("U??ivate??")
        chatbot = vg_user("Chatbot")

        intent = comp("Intent Klasifik??tor")
        entity = comp("Ekstrahovanie Entity")
        action = comp("Action")

        nlp = VGroup(intent,entity).arrange(RIGHT)
        NLP = SurroundingRectangle(nlp,buff=MED_SMALL_BUFF)
        NLP_title = Text("NLP Preprocess").next_to(NLP,UP)

        pipeline = VGroup(VGroup(NLP,nlp,NLP_title),action).arrange_in_grid(rows=2,cols=1,buff=1)

        whole = VGroup(user,pipeline,chatbot).arrange(RIGHT,buff=2)
        self.play(Write(whole,run_time=5))

        user_NLP = Arrow(user,NLP.get_left(),stroke_width=5)
        NLP_action = Arrow(NLP.get_bottom(),action.get_top(),stroke_width=5)
        action_chatbot = Arrow(action.get_right(),chatbot.get_left(),stroke_width=5 )
        arrows = VGroup(user_NLP,NLP_action,action_chatbot)
        for arrow in arrows:
            self.play(GrowArrow(arrow))

        self.play(FadeOut(dialog, shift=UP))
        last,dialog = Overview.dialog(self,2, init, "U??ivate?? po??le spr??vu")

        u_msg = message(Colors.teal_e.value,WHITE,"Hello").next_to(user,2*DOWN)

        self.play(ReplacementTransform(user.copy(),u_msg))
        self.play(ReplacementTransform(u_msg.copy(),user_NLP))
        #'''
        self.play(FadeOut(dialog, shift=UP))

        last,dialog = Overview.dialog(self,2, init, "Spr??va sa po??le do pipelinu")

        self.play(ReplacementTransform(user_NLP.copy(),NLP))
        self.play(ReplacementTransform(NLP.copy(),NLP_action))
        self.play(ReplacementTransform(NLP_action.copy(),action))
        self.play(ReplacementTransform(action.copy(),action_chatbot))
        self.play(ReplacementTransform(action_chatbot.copy(),chatbot))

        self.play(FadeOut(dialog, shift=UP))
        last,dialog = Overview.dialog(self,2, init, "Chatbot odpoved??")
        self.play(FadeOut(dialog, shift=UP))

        c_msg = message(Colors.gray.value,WHITE,"Hi").next_to(user,2*DOWN).next_to(chatbot,2*DOWN)
        self.play(ReplacementTransform(chatbot.copy(),c_msg))
        self.wait()

        last,dialog = Overview.dialog(self,1, init, "??o je NLP procesovanie?")
        self.play(FadeOut(dialog, shift=UP))
        last,dialog = Overview.dialog(self,1, init, "??o je Intent a Entita?")
        self.play(FadeOut(dialog, shift=UP))

        last,dialog = Overview.dialog(self,2, init, "Za??neme v nejakom porad??")

        self.play(Unwrite(user),Unwrite(chatbot),Unwrite(NLP),Unwrite(arrows),Unwrite(u_msg),Unwrite(c_msg),Unwrite(intent[0]),Unwrite(entity[0]),Unwrite(action[0]))

        list = VGroup(NLP_title,intent[1],entity[1],action[1])
        self.play(list.animate.arrange(DOWN,buff=1))
        self.play(list.animate.shift(4*LEFT))
        self.wait()
        self.play(FadeOut(dialog, shift=UP))
        last,dialog = Overview.dialog(self,2, init, "Najprv NLP procesovanie")
        self.play(Indicate(NLP_title))
        self.play(FadeOut(dialog, shift=UP))
        self.play(Unwrite(intent[1]),Unwrite(entity[1]),Unwrite(action[1]))
        self.play(NLP_title.animate.shift(4*RIGHT).scale(1.25))
        self.wait()
        self.play(FadeOut(NLP_title,shift=UP))
        self.play(*[FadeOut(mob)for mob in self.mobjects])


class Chat(Scene):

    def dialog(self, person, last, string):
        if person == 1:
            dialog = message(Colors.teal_e.value,WHITE,string).next_to(last,DOWN+2*LEFT)
            dots = Text("...").move_to(dialog.get_center())
            self.play(FadeIn(dots))
            self.play(Unwrite(dots))
            self.play(FadeIn(dialog, shift=UP))
            self.wait()
            last = last.move_to(last.get_center()+DOWN)
            return last
        else:
            dialog = message(Colors.gray.value,WHITE,string).next_to(last,DOWN+2*RIGHT)
            dots = Text("...").move_to(dialog.get_center())
            self.play(FadeIn(dots))
            self.play(Unwrite(dots))
            self.play(FadeIn(dialog, shift=UP))
            self.wait()
            last = last.move_to(last.get_center()+DOWN)
            return last

    def construct(self):
            bg_color = "#001514"
            self.camera.background_color = bg_color
            Text.set_default(font_size=25)
            text = Text("Ako funguje ChatBot")
            self.play(Write(text))
            self.wait()
            self.play(Unwrite(text))

            init = Rectangle(width=0).move_to(4*UP)
            last = Chat.dialog(self,1, init, "Ahoj. ??o je Chatbot?")
            last = Chat.dialog(self,1, last, "Ako funguje Chatbot?")
            last = Chat.dialog(self,2, last, "Po??kaj, spoma??")
            last = Chat.dialog(self,2, last, "Uk????eme si najprv v??etko\n a n??sledne si prejdeme\n ??as?? po ??as??")
            last = Chat.dialog(self,1, last, "OK :)")

            self.play(*[FadeOut(mob)for mob in self.mobjects])
