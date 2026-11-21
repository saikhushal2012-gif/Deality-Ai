import unittest

from comparewise.project_qa import answer_project_question


class ProjectQATests(unittest.TestCase):
    def test_savings_answer_does_not_claim_measured_results(self):
        answer = answer_project_question("How much money does Deality AI save?")
        self.assertIn("not measured", answer)
        self.assertIn("potential benefit", answer)

    def test_ai_answer_distinguishes_gemini_from_deterministic_recommendations(self):
        answer = answer_project_question("Does your chat assistant use Gemini?")
        self.assertIn("live recommendation score is deterministic", answer)
        self.assertIn("can use Gemini", answer)

    def test_unknown_project_question_does_not_invent_a_fact(self):
        answer = answer_project_question("What is the app's current adoption rate?")
        self.assertIn("don't have a verified project-specific answer", answer)
        self.assertIn("live recommendation", answer)


if __name__ == "__main__":
    unittest.main()