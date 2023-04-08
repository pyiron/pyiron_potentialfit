from unittest import TestCase, skipUnless
from sys import version_info

from pyiron_contrib.workflow.channels import InputChannel, OutputChannel


class DummyNode:
    def update(self):
        pass


@skipUnless(version_info[0] == 3 and version_info[1] >= 10, "Only supported for 3.10+")
class TestChannels(TestCase):

    def setUp(self) -> None:
        self.ni1 = InputChannel(label="numeric", node=DummyNode(), default=1, type_hint=int | float)
        self.ni2 = InputChannel(label="numeric", node=DummyNode(), default=1, type_hint=int | float)
        self.no = OutputChannel(label="numeric", node=DummyNode(), default=0, type_hint=int | float)

        self.so1 = OutputChannel(label="list", node=DummyNode(), default=["foo"], type_hint=list)
        self.so2 = OutputChannel(label="list", node=DummyNode(), default=["foo"], type_hint=list)

    def test_mutable_defaults(self):
        self.so1.default.append("bar")
        self.assertEqual(
            len(self.so2.default),
            len(self.so1.default) - 1,
            msg="Mutable defaults should avoid sharing between instances"
        )

    def test_connections(self):

        with self.subTest("Test connection reflexivity and value updating"):
            self.assertEqual(self.no.value, 0)
            self.ni1.connect(self.no)
            self.assertIn(self.no, self.ni1.connections)
            self.assertIn(self.ni1, self.no.connections)
            self.assertEqual(self.no.value, self.ni1.value)

        with self.subTest("Test disconnection"):
            self.ni2.disconnect(self.no)  # Should do nothing
            self.ni1.disconnect(self.no)
            self.assertEqual(
                [], self.ni1.connections, msg="No connections should be left"
            )
            self.assertEqual(
                [],
                self.no.connections,
                msg="Disconnection should also have been reflexive"
            )

        with self.subTest("Test multiple connections"):
            self.no.connect(self.ni1, self.ni2)
            self.assertEqual(2, len(self.no.connections), msg="Should connect to all")

        with self.subTest("Test iteration"):
            self.assertTrue(all([con in self.no.connections for con in self.no]))

    def test_connection_validity_tests(self):
        self.ni1.type_hint = int | float | bool  # Override with a larger set
        self.ni2.type_hint = int  # Override with a smaller set

        with self.assertRaises(TypeError):
            self.ni1.connect("Not a channel at all")

        self.no.connect(self.ni1)
        self.assertIn(
            self.no,
            self.ni1.connections,
            "Input types should be allowed to be a super-set of output types"
        )

        self.no.connect(self.ni2)
        self.assertNotIn(
            self.no,
            self.ni2.connections,
            "Input types should not be allowed to be a sub-set of output types"
        )

        self.so1.connect(self.ni2)
        self.assertNotIn(
            self.so1,
            self.ni2.connections,
            "Totally different types should not allow connections"
        )

        self.ni2.strict_connections = False
        self.so1.connect(self.ni2)
        self.assertIn(
            self.so1,
            self.ni2.connections,
            "With strict connections turned off, we should allow type-violations"
        )

    def test_ready(self):
        self.no.value = 1
        self.assertTrue(self.no.ready)
        self.no.value = "Not numeric at all"
        self.assertFalse(self.no.ready)

    def test_update(self):
        self.no.connect(self.ni1, self.ni2)
        self.no.update(42)
        for inp in self.no.connections:
            self.assertEqual(
                self.no.value,
                inp.value,
                msg="Value should have been passed downstream"
            )
