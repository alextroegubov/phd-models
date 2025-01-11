from utils import ParametersSet
import itertools
import logging
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Callable, List
import random
from enum import Enum


import numpy as np
from typing import Optional
from functools import partial
from abc import abstractmethod, ABC

test_params = ParametersSet(2, [15, 3], [1, 1], [1, 5], 1, 50, 1, 1, 50)


class EventType(Enum):
    REQUEST_ARRIVAL = 1
    REQUEST_SERVICE = 2
    STOP_SIMULATION = 3


@dataclass(order=True)
class Event:
    time: float
    action: Callable[[], None] | None = field(compare=False)  # event action
    event_type: EventType = field(compare=False)
    event_id: int = field(compare=False)  # Optional identifier for debugging


class Simulator:
    # singleton class
    CANCELED_EVENT_TIME = -100
    _instance = None
    num_instances = 0

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            Simulator._instance = Simulator()
        return Simulator._instance

    def __init__(self):

        assert (
            self.__class__.num_instances == 0
        ), "Simulator class already exists somewhere"

        self.current_time = 0.0
        self.event_id_counter = 0
        self.event_queue = PriorityQueue()
        self.event_lookup: dict[int, Event] = {}

        self.processed_events = 0

        self.__class__.num_instances += 1

    def _get_next_id(self):
        new_id = self.event_id_counter
        self.event_id_counter += 1
        return new_id

    def schedule_event(
        self, time: float, action: Callable[[], None], event_type: EventType
    ) -> int:
        """Schedules an event at a specific simulation time."""
        event_id = self._get_next_id()
        event = Event(time, action, event_type, event_id)

        self.event_lookup[event_id] = event
        self.event_queue.put(event)
        return event_id

    def cancel_event(self, event_id: int):
        event = self.event_lookup.get(event_id)
        if event is not None:
            # delete from lookup table
            del self.event_lookup[event_id]
            # make field invalid for queue
            event.time = self.CANCELED_EVENT_TIME
            event.action = None  # lambda *args, **kwargs: None
        else:
            print("Cancel non-existing event")

    def run(self):
        """Starts the simulation loop."""
        while not self.event_queue.empty():
            event = self.event_queue.get()
            if event.time != self.CANCELED_EVENT_TIME:
                self.current_time = event.time
                event.action()
                self.processed_events += 1

    def get_current_time(self) -> float:
        """Returns the current simulation time."""
        return self.current_time

    def stop(self):
        """Stops the simulation."""
        pass


class Request:
    def __init__(
        self,
        request_id: int,
        flow_id: int,
        arrival_time: float,
        total_size: float,
        min_resources: int,
    ):
        """
        Initialize a request.

        :param flow_id: ID of the flow generating the request
        :param arrival_time: Time at which the request arrives
        :param resource_required: Amount of network resource required for this request
        """
        # request_id
        self.request_id = request_id
        # flow id
        self.flow_id = flow_id
        # time the request is generated and sent to network
        self.arrival_time = arrival_time
        # min_resources * service_time_with_min_resources, total request_size
        self.total_size = total_size
        # TBD: remove according to logic?
        self.min_resources = min_resources

        # served request size
        self.served_size = 0
        # last time current_alloc_resource changed
        self.last_alloc_time = self.arrival_time

        self.current_alloc_resource = 0
        self.total_alloc_resources = 0

        self.end_service_time = 0
        self.service_event_id = -1

    def _update_served_size_and_last_alloc_time(self):
        time = Simulator.get_instance().get_current_time()
        served_size = (time - self.arrival_time) * self.current_alloc_resource
        assert served_size >= 0, f"negative {served_size}"

        self.served_size += served_size
        self.last_alloc_time = time

    def realloc_resources(self, new_alloc_resources) -> float:
        self._update_served_size_and_last_alloc_time()
        self.current_alloc_resources = new_alloc_resources
        left_for_service = self.total_size - self.served_size
        service_time_left = left_for_service / self.current_alloc_resources
        assert service_time_left >= 0, f"negative {service_time_left}"

        return service_time_left


class Flow(ABC):
    def __init__(
        self,
        flow_id: int,
        arrival_rate: float,
        service_rate: float,
        on_arrival_callback: Optional[Callable[[Request], None]] = None,
    ):
        """
        Initialize flow parameters and store statistics.
        :param flow_id: Unique identifier for the flow
        :param lambda_rate: Request generation rate (1 / average inter-arrival time)
        :param service_rate: Request service rate (1 / average service time)
        :param fixed_resource: Fixed resource allocation for requests (if any)
        :param dynamic_resource_range: Tuple (b_min, b_max) for dynamic resource allocation
        """
        self.flow_id = flow_id
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.on_arrival_callback = on_arrival_callback

        self.generated_requests = 0

    def set_on_arrival_callback(self, fn: Callable[[Request], None]):
        self.on_arrival_callback = fn

    def generate_next_arrival_time(self) -> float:
        """Generates the time until the next request using exponential distribution."""
        return random.expovariate(self.arrival_rate)  # 1 / (1 / lambda)

    def generate_request(self):
        size = self.generate_service_time() * self._get_min_resources()
        request = Request(
            flow_id=self.flow_id,
            arrival_time=Simulator.get_instance().get_current_time(),
            total_size=size,
            min_resources=self._get_min_resources(),
        )

        self.on_arrival_callback(request)
        self._schedule_request()

    def _schedule_request(self):
        Simulator.get_instance().schedule_event(
            time=self.generate_next_arrival_time()
            + Simulator.get_instance().get_current_time(),
            action=self.generate_request,
            event_type=EventType.REQUEST_ARRIVAL,
        )

    def start_flow(self):
        self._schedule_request()

    def request_accepted(self):
        pass

    def request_rejected(self):
        pass

    def request_serviced(self, req: Request):
        pass

    @abstractmethod
    def _get_min_resources(self) -> int:
        pass

    @abstractmethod
    def _generate_service_time(self) -> float:
        pass


class RealTimeFlow(Flow):
    def __init__(
        self,
        flow_id: int,
        arrival_rate: float,
        service_rate: float,
        fixed_resource: int,
    ):
        super().__init__(flow_id, arrival_rate, service_rate)
        self.fixed_resources = fixed_resource

    def _get_min_resources(self) -> int:
        return self.fixed_resource

    def _generate_service_time(self) -> float:
        """Generates the service time using exponential distribution."""
        return random.expovariate(self.service_rate)


class ElasticDataFlow(Flow):
    def __init__(
        self,
        flow_id: int,
        arrival_rate: float,
        service_rate: float,
        min_resources: int,
        max_resources: int,
    ):
        super().__init__(flow_id, arrival_rate, service_rate)
        self.min_resources = min_resources
        self.max_resources = max_resources

    def _get_min_resources(self) -> int:
        return self.min_resources

    def _generate_service_time(self) -> float:
        pass
        """Generates the service time using exponential distribution."""
        return random.expovariate(self.service_rate)


class Network:
    def __init__(self, params: ParametersSet):
        """
        Initialize network parameters.
        :param capacity: Total network resource capacity (V)
        """
        self.params: ParametersSet = params
        self.capacity = self.params.beam_capacity
        self.requests_lookup = {}
        self.flows_lookup = {}
        self.total_allocated = 0

    def add_flows(self, n_real_time_flows: int, n_elastic_flows: int):
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
                self.params.data_resources_min,
                self.params.data_resources_max,
            )
            self.flows_lookup[flow_id] = flow
            flow_id += 1

        for _, flow in self.flows_lookup.items():
            flow.set_on_arrival_callback(self.request_arrival_handler)
            flow.start_flow()

    def request_arrival_handler(self, request: Request):
        # check if request can be served
        min_alloc_resources = 0
        for _, request in self.requests_lookup.items():
            min_alloc_resources += request.min_resources

        if request.min_resources + min_alloc_resources <= self.capacity:
            self.accept_request(request)
            self.flows_lookup[request.flow_id].request_accepted()
        else:
            self.flows_lookup[request.flow_id].request_rejected()

    def accept_request(self, request: Request):
        its_flow = self.flows_lookup[request.flow_id]
        self.requests_lookup[request.request_id] = request

        # TBD refactor?
        if isinstance(its_flow, RealTimeFlow):
            request.realloc_resources(its_flow.fixed_resources)
        elif isinstance(its_flow, ElasticDataFlow):
            pass

        self.redistribute_elastic_resources()

    def redistribute_elastic_resources(self):
        """
        Dynamically adjust resource allocation among requests.
        """
        total_min_alloc = sum(
            req.min_resources for req in self.requests_lookup.values()
        )
        assert total_min_alloc <= self.capacity

        real_time_requests = [
            req
            for req in self.requests_lookup.values()
            if isinstance(self.flows_lookup[req.flow_id], RealTimeFlow)
        ]
        real_time_resources = sum(req.alloc_resources for req in real_time_requests)
        elastic_data_requests = [
            req
            for req in self.requests_lookup.values()
            if isinstance(self.flows_lookup[req.flow_id], ElasticDataFlow)
        ]
        resources_left_for_elastic = self.capacity - real_time_resources


        total_allocated = real_time_resources

        round_down_resources = resources_left_for_elastic // len(elastic_data_requests)
        num_request_with_round_up = resources_left_for_elastic % len(
            elastic_data_requests
        )
        num_request_with_round_down = (
            len(elastic_data_requests) - num_request_with_round_up
        )

        assert (
            resources_left_for_elastic
            == round_down_resources * num_request_with_round_down
            + (round_down_resources + 1) * num_request_with_round_up
        )
        # allocate resources for requests
        # TBD: check min and max possible according to flow
        for req in elastic_data_requests:
            its_flow = self.flows_lookup[req.flow_id]
            suggested_resources = round_down_resources + 1 if num_request_with_round_up > 0 else round_down_resources
            alloc_resources = min(suggested_resources, its_flow.max_resources)

            new_service_time = req.realloc_resources(alloc_resources)
            total_allocated += alloc_resources
            Simulator.get_instance().cancel_event(req.service_event_id)
            Simulator.get_instance().schedule_event(
                time=Simulator.get_instance().get_current_time() + new_service_time,
                action=partial(self.service_request, req.request_id),
            )



    def request_service_handler(self, req_id: int):
        request = self.requests_lookup[req_id]
        request.end_service_time = Simulator.get_instance().get_current_time()
        request.realloc_resources(0)

        assert np.isclose(request.served_size, request.total_size), f"{request.served_size} != {request.total_size}"
        self.flows_lookup[request.flow_id].request_serviced(request)

