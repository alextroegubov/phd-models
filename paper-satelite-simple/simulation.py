"""Simulation module for the satellite network model."""

import logging
from queue import PriorityQueue
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from abc import abstractmethod, ABC

from typing import Callable, Optional

import random
import numpy as np
from utils import ParametersSet

logger = logging.getLogger(__name__)
logging.basicConfig(filename="simulation.log", filemode="w", level=logging.DEBUG, encoding="utf-8")


test_params = ParametersSet(2, [15, 3], [1, 1], [1, 5], 1, 50, 1, 1, 50)


class EventType(Enum):
    """Enumeration for event types."""

    REQUEST_ARRIVAL = 1
    REQUEST_SERVICE = 2
    STOP_SIMULATION = 3
    CANCELLED = 4


@dataclass(order=True)
class Event:
    """Class for simulation events.

    :param time: Time at which the event occurs
    :param action: Function to be executed when the event occurs
    :param event_type: Type of the event
    :param event_id: Unique identifier for the event
    """

    time: float
    action: Callable[[], None] = field(compare=False)
    event_type: EventType = field(compare=False)
    event_id: int = field(compare=False)


class Simulator:
    """Simulator class for event simulation. Singleton class"""

    @staticmethod
    def cancelled_event_action(*args, **kwargs):
        """Action placeholder for cancelled events."""
        raise AttributeError("Cancelled event executed")

    CANCELED_EVENT_TIME = -100
    # def __new__(cls):
    #     if not hasattr(cls, "_instance"):
    #         cls._instance = super(Simulator, cls).__new__(cls)
    #     return cls._instance
    _instance = None
    num_instances = 0

    @classmethod
    def get_instance(cls):
        """Get instance of the simulator or create it if it does not exist."""
        if cls._instance is None:
            Simulator._instance = Simulator()
        return Simulator._instance

    @classmethod
    def schedule_event(cls, time: float, action: Callable[[], None], e_type: EventType) -> int:
        """Schedules an event at a specific simulation time.

        :param time: Time at which the event occur
        :param action: Function to be executed
        :param event_type: Type of the event
        :return: Unique identifier for the event
        """
        instance = cls.get_instance()
        event_id = instance._get_next_id()
        event = Event(time, action, e_type, event_id)
        instance.event_lookup[event_id] = event
        instance.event_queue.put(event)

        logger.debug(
            "%s Event[id=%s, type=%s] scheduled at %s",
            instance.get_current_time(),
            event_id,
            e_type,
            time,
        )
        return event_id

    @classmethod
    def cancel_event(cls, event_id: int):
        """Cancels a scheduled event.

        :param event_id: Unique identifier for the event
        """
        instance = cls.get_instance()
        event = instance.event_lookup.get(event_id)
        if event is not None:
            # delete from lookuptable
            del instance.event_lookup[event_id]
            # make fields invalid
            event.time = instance.CANCELED_EVENT_TIME
            event.type = EventType.CANCELLED
            event.action = cls.cancelled_event_action

            logger.debug("%s Event[id=%s] cancelled", instance.get_current_time(), event_id)
        else:
            logger.warning(
                "%s Event[id=%s] cannot be cancelled", instance.get_current_time(), event_id
            )

    @classmethod
    def run(cls, number_of_events: int):
        """Starts the simulation loop."""
        instance = cls.get_instance()

        while not instance.event_queue.empty() or instance.processed_events < number_of_events:
            event = instance.event_queue.get()
            if event.type is not EventType.CANCELLED:
                instance.current_time = event.time
                logger.debug(
                    "%s Event[id=%s] executed", instance.get_current_time(), event.event_id
                )
                event.action()
                instance.processed_events += 1

        if instance.event_queue.empty():
            logger.warning("%s Simulation ended: empty event queue", instance.get_current_time())
        else:
            logger.info(
                "%s Simulation ended: processed %s events",
                instance.get_current_time(),
                instance.processed_events,
            )

    @classmethod
    def get_current_time(cls) -> float:
        """Return current simulation time."""
        instance = cls.get_instance()
        return instance.current_time

    def __init__(self):

        if self.__class__.num_instances > 0:
            logger.error("Simulator class already exists somewhere")
        self.__class__.num_instances += 1

        self.current_time: float = 0.0

        self.event_id_counter: int = 0
        self.event_queue = PriorityQueue()
        self.event_lookup: dict[int, Event] = {}

        self.processed_events: int = 0

    def _get_next_id(self):
        new_id = self.event_id_counter
        self.event_id_counter += 1
        return new_id


class Request:
    """Class for requests in the network.d"""

    next_req_id = 0

    def __init__(
        self,
        flow_id: int,
        arrival_time: float,
        total_size: float,
        min_resources: int,
    ):
        """Initialize a request.

        :param flow_id: Unique identifier of flow created the request
        :param arrival_time: Time at which the request is generated
        :param min_resources: Minimum resources required to serve the request
        :param service_event_id: Unique identifier for the service event
        :param total_size: Total size of the request [resource_units * time]
        :param served_size: Already served size of the request
        :param last_alloc_time: Time at which the last resource allocation is made
        :param current_alloc_resource: Current resource allocation
        :param end_service_time: Time at which the request is served
        """
        self.request_id: int = self.__class__.next_req_id
        self.__class__.next_req_id += 1

        self.flow_id: int = flow_id
        self.arrival_time: float = arrival_time
        # TBD: remove according to logic?
        self.min_resources: int = min_resources
        self.service_event_id = -1

        self.total_size: float = total_size
        self.served_size: float = 0

        self.last_alloc_time: float = self.arrival_time
        self.current_alloc_resource: int = 0
        self.end_service_time: float = 0.0

    def _update_served_size_and_last_alloc_time(self):
        """Updates the served size and last allocation time of the request.
        Does not change the current allocation resources.
        """
        time = Simulator.get_instance().get_current_time()
        served_size = (time - self.arrival_time) * self.current_alloc_resource
        self.served_size += served_size
        self.last_alloc_time = time

        if served_size < 0:
            logger.error(
                "%s request[id=%s] has negative served size: %s", time, self.request_id, served_size
            )

    def realloc_resources(self, new_alloc_resources: int) -> float:
        """Reallocate resources for the request and return the remaining service time.
        :param new_alloc_resources: New resource allocation for the request
        :return: Remaining service time for the request
        """
        self._update_served_size_and_last_alloc_time()
        self.current_alloc_resource = new_alloc_resources
        service_time_left = (self.total_size - self.served_size) / self.current_alloc_resource

        if service_time_left < 0:
            logger.error(
                "%s request[id=%s] has negative service time left: %s",
                Simulator.get_current_time(),
                self.request_id,
                service_time_left,
            )

        return service_time_left


class Flow(ABC):
    """Class for modeling traffic flows in the network."""

    def __init__(
        self,
        flow_id: int,
        arrival_rate: float,
        service_rate: float,
        on_arrival_callback: Optional[Callable[[Request], None]] = None,
    ):
        """Initialize flow parameters and store statistics.

        :param flow_id: Unique identifier for the flow
        :param arrival_rate: Request generation rate (1 / average inter-arrival time)
        :param service_rate: Request service rate (1 / average service time)
        :param on_arrival_callback: Callback function to be executed when a request arrives
        """
        self.flow_id = flow_id
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.on_arrival_callback = on_arrival_callback
        self.is_active = False

        # some statistics
        self.total_gen_requests = 0
        self.total_accepted_requests = 0
        self.total_serviced_requests = 0
        self.total_rejected_requests = 0

        self.actual_service_time: list[float] = []
        self.actual_mean_resources: list[float] = []

    def set_on_arrival_callback(self, fn: Callable[[Request], None]):
        """Set the callback function to be executed when a request arrives."""
        self.on_arrival_callback = fn

    def generate_next_arrival_time(self) -> float:
        """Generates the time until the next request using exponential distribution."""
        return random.expovariate(self.arrival_rate)  # 1 / (1 / arrival_intensity)

    def generate_request(self):
        """Generates a new request and schedules the next request arrival."""
        size = self._generate_service_time() * self._get_min_resources()
        req = Request(
            flow_id=self.flow_id,
            arrival_time=Simulator.get_current_time(),
            total_size=size,
            min_resources=self._get_min_resources(),
        )

        logger.debug(
            "%s Request[id=%s, flow_id=%s, size=%s] generated",
            Simulator.get_current_time(),
            req.request_id,
            req.flow_id,
            req.total_size,
        )

        self.total_gen_requests += 1
        self.on_arrival_callback(req)
        self._schedule_request()

    def _schedule_request(self):
        """Schedules the request arrival event."""
        arrival_time = self.generate_next_arrival_time() + Simulator.get_current_time()
        Simulator.schedule_event(
            time=arrival_time,
            action=self.generate_request,
            e_type=EventType.REQUEST_ARRIVAL,
        )

    def start_flow(self):
        """Starts the flow by generating the first request."""
        if not self.is_active:
            self.is_active = True
            self._schedule_request()
        else:
            logger.warning(
                "%s Flow[id=%s] is already active", Simulator.get_current_time(), self.flow_id
            )

    def request_accepted(self):
        """Increments the number of accepted requests."""
        self.total_accepted_requests += 1

    def request_rejected(self):
        """Increments the number of rejected requests."""
        self.total_rejected_requests += 1

    def request_serviced(self, req: Request):
        """Increments the number of serviced requests and stores the QoS stats."""
        self.total_serviced_requests += 1
        service_time = req.end_service_time - req.arrival_time

        self.actual_service_time.append(service_time)
        self.actual_mean_resources.append(req.size / service_time)

    @abstractmethod
    def _get_min_resources(self) -> int:
        pass

    @abstractmethod
    def _generate_service_time(self) -> float:
        pass


class RealTimeFlow(Flow):
    """Class for real-time data flows in the network."""

    def __init__(
        self,
        flow_id: int,
        arrival_rate: float,
        service_rate: float,
        fixed_resource: int,
    ):
        super().__init__(flow_id, arrival_rate, service_rate)
        self.fixed_resource = fixed_resource

    def _get_min_resources(self) -> int:
        """Return minimum resources required to serve the request."""
        return self.fixed_resource

    def _generate_service_time(self) -> float:
        """Generates the service time using exponential distribution."""
        return random.expovariate(self.service_rate)


class ElasticDataFlow(Flow):
    """Class for elastic data flows in the network."""

    def __init__(
        self,
        flow_id: int,
        arrival_rate: float,
        service_rate: float,
        min_max_resources_range: tuple[int, int],
    ):
        super().__init__(flow_id, arrival_rate, service_rate)
        self.min_resources, self.max_resources = min_max_resources_range

    def _get_min_resources(self) -> int:
        """Return minimum resources required to serve the request."""
        return self.min_resources

    def _generate_service_time(self) -> float:
        """Generates the service time using exponential distribution."""
        return random.expovariate(self.service_rate)


class Network:
    """Class for the network model."""

    def __init__(self, params: ParametersSet):
        """Initialize network parameters.
        :param params: Network parameters
        """
        self.params: ParametersSet = params
        self.capacity = self.params.beam_capacity

        self.requests_lookup: dict[int, Request] = {}
        self.flows_lookup: dict[int, Flow] = {}

    def add_flows(self, n_real_time_flows: int, n_elastic_flows: int):
        """Add real-time data and elastic data flows to the network."""
        flow_id = 0

        for i in range(n_real_time_flows):
            flow = RealTimeFlow(
                flow_id,
                self.params.real_time_lambdas[i],
                self.params.real_time_mus[i],
                self.params.real_time_resources[i],
            )
            self.flows_lookup[flow_id] = flow
            flow_id += 1

        for _ in range(n_elastic_flows):
            flow = ElasticDataFlow(
                flow_id,
                self.params.data_lambda,
                self.params.data_mu,
                (self.params.data_resources_min, self.params.data_resources_max),
            )
            self.flows_lookup[flow_id] = flow
            flow_id += 1

        if n_real_time_flows + n_elastic_flows != len(self.flows_lookup):
            logger.error(
                "Flows not added correctly: rt flows = %s, data flows = %s, total = %s",
                n_real_time_flows,
                n_elastic_flows,
                len(self.flows_lookup),
            )

        for _, flow in self.flows_lookup.items():
            flow.set_on_arrival_callback(self.request_arrival_handler)
            flow.start_flow()

    def request_arrival_handler(self, new_req: Request):
        """Handle the arrival of a new request: acept or reject."""
        # required resources for existing requests
        min_required_resources = sum(req.min_resources for req in self.requests_lookup.values())
        its_flow = self.flows_lookup[new_req.flow_id]

        if new_req.min_resources + min_required_resources <= self.capacity:
            self.accept_request(new_req)
            its_flow.request_accepted()
        else:
            its_flow.request_rejected()

    def accept_request(self, new_req: Request):
        """Process accepting a new request."""
        its_flow = self.flows_lookup[new_req.flow_id]
        self.requests_lookup[new_req.request_id] = new_req

        if isinstance(its_flow, RealTimeFlow):
            new_req.realloc_resources(its_flow.fixed_resources)

        self.redistribute_elastic_resources()

    def redistribute_elastic_resources(self):
        """Dynamically adjust resource allocation among elastic requests."""
        min_required_resources = sum(req.min_resources for req in self.requests_lookup.values())
        if min_required_resources > self.capacity:
            logger.error(
                "Total required resources exceed capacity: %s > %s",
                min_required_resources,
                self.capacity,
            )

        real_time_requests = [
            req
            for req in self.requests_lookup.values()
            if isinstance(self.flows_lookup[req.flow_id], RealTimeFlow)
        ]
        elastic_data_requests = [
            req
            for req in self.requests_lookup.values()
            if isinstance(self.flows_lookup[req.flow_id], ElasticDataFlow)
        ]

        real_time_resources = sum(req.alloc_resources for req in real_time_requests)
        resources_left_for_elastic = self.capacity - real_time_resources

        already_allocated = real_time_resources

        # resources per request rounded down: q
        round_down_resources = resources_left_for_elastic // len(elastic_data_requests)
        # number of requests with (q + 1) resources
        num_request_with_round_up = resources_left_for_elastic % len(elastic_data_requests)
        # number of requests with (q) resources
        num_request_with_round_down = len(elastic_data_requests) - num_request_with_round_up

        if resources_left_for_elastic != round_down_resources * num_request_with_round_down + (round_down_resources + 1) * num_request_with_round_up:
            

        assert (
            resources_left_for_elastic
            == round_down_resources * num_request_with_round_down
            + (round_down_resources + 1) * num_request_with_round_up
        )
        # allocate resources for requests
        # TBD: check min and max possible according to flow
        for req in elastic_data_requests:
            its_flow = self.flows_lookup[req.flow_id]
            suggested_resources = (
                round_down_resources + 1 if num_request_with_round_up > 0 else round_down_resources
            )
            alloc_resources = min(suggested_resources, its_flow.max_resources)

            new_service_time = req.realloc_resources(alloc_resources)
            already_allocated += alloc_resources
            Simulator.get_instance().cancel_event(req.service_event_id)
            Simulator.get_instance().schedule_event(
                time=Simulator.get_instance().get_current_time() + new_service_time,
                action=partial(self.request_service_handler, req.request_id),
            )

    def request_service_handler(self, req_id: int):
        request = self.requests_lookup[req_id]
        request.end_service_time = Simulator.get_instance().get_current_time()
        request.realloc_resources(0)

        assert np.isclose(
            request.served_size, request.total_size
        ), f"{request.served_size} != {request.total_size}"
        self.flows_lookup[request.flow_id].request_serviced(request)
