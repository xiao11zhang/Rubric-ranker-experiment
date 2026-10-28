import unittest
from .local_repair_specs import scoped
from .entities import transform


class RepairScopeTest(unittest.TestCase):
    def test_place_excludes_institution_and_preserves_relations(self):
        source={'description':'London hosts the University of London; London is not Paris in 1990.'}
        city=scoped({'type':'location','mentions':['London']},source,['University of London'])
        result,_,changes=transform(source,[city,{'type':'organization','mentions':['University of London']}])
        self.assertEqual(result['description'],'<PLACE_A> hosts the <ORG_A>; <PLACE_A> is not Paris in 1990.')
        self.assertEqual(len(changes),3)

    def test_full_and_short_alias_keep_one_placeholder(self):
        source={'description':'John Smith met John in York; Smith University is separate.'}
        person=scoped({'type':'person','mentions':['John Smith','John']},source,['Smith University'])
        result,_,_=transform(source,[person])
        self.assertEqual(result['description'],'<PERSON_A> met <PERSON_A> in York; Smith University is separate.')


if __name__ == '__main__': unittest.main()
