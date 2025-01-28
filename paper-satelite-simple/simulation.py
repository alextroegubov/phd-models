"""Simulation module for the satellite network model."""

from __future__ import annotations
from typing import Callable, Optional

import logging
from queue import PriorityQueue
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from abc import abstractmethod, ABC
import time
import argparse

import random
import numpy as np
from utils import ParametersSet, Metrics

logger = logging.getLogger(__name__)
logging.basicConfig(filename="simulation.log", filemode="w", level=logging.DEBUG, encoding="utf-8")


class StatsBaseClass(ABC):
    """Base class for statistics collection."""

    def __init__(self):
        self.stats_enabled = False

    def enable_stats_collection(self):
        """Enable statistics collection."""
        self.stats_enabled = True
        self._start_stats_collection()

    @staticmethod
    def if_stats_enabled(method):
        """Decorator to enable statistics collection for a method."""

        def wrapper(self, *args, **kwargs):
            if self.stats_enabled:
                return method(self, *args, **kwargs)

            return None

        return wrapper

    @abstractmethod
    def _start_stats_collection(self):
        pass

    @abstractmethod
    def get_stats(self):
        pass


class EventType(Enum):
    """Enumeration for event types."""

    REQUEST_ARRIVAL = 1
    REQUEST_SERVICE = 2
    STOP_SIMULATION = 3
    CANCELLED = 4


@dataclass(order=True)
class Event:
    """Class for simulation events.

    :param time_at: Time at which the event occurs
    :param action: Function to be executed when the event occurs
    :param type: Type of the event
    :param event_id: Unique identifier for the event
    """

    time_at: float
    action: Callable[[], None] = field(compare=False)
    type: EventType = field(compare=False)
    event_id: int = field(compare=False)


class Simulator:
    """Simulator class for event simulation. Singleton class"""

    @staticmethod
    def cancelled_event_action(*args, **kwargs):
        """Action placeholder for cancelled events."""
        raise AttributeError("Cancelled event executed")

    _instance: Optional[Simulator] = None
    num_instances = 0

    @classmethod
    def get_instance(cls) -> Simulator:
        """Get instance of the simulator or create it if it does not exist."""
        if cls._instance is None:
            cls._instance = Simulator()
        return cls._instance

    @classmethod
    def schedule_event(cls, time_at: float, action: Callable[[], None], e_type: EventType) -> int:
        """Schedules an event at a specific simulation time.

        :param time_at: Time at which the event occur
        :param action: Function to be executed
        :param e_type: Type of the event
        :return: Unique identifier for the event
        """
        instance = cls.get_instance()
        event_id = instance._get_next_id()
        event = Event(time_at, action, e_type, event_id)
        instance.event_lookup[event_id] = event
        instance.event_queue.put(event, block=False)

        logger.debug(
            "%4.5f Event[id=%s, type=%s] scheduled at %4.5f; queue_size=%s",
            instance.get_current_time(),
            event_id,
            e_type.name,
            time_at,
            instance.event_queue.qsize(),
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
            logger.debug(
                "%4.5f Event[id=%s, type=%s, time=%4.5f] cancelled",
                instance.get_current_time(),
                event_id,
                event.type.name,
                event.time_at,
            )
            # make fields invalid
            event.type = EventType.CANCELLED
            event.action = cls.cancelled_event_action
        else:
            logger.warning("%4.5f No Event[id=%s] to cancel", instance.get_current_time(), event_id)

    @classmethod
    def run(cls, number_of_events: int):
        """Starts the simulation loop.

        :param number_of_events: Number of events to process
        """
        instance = cls.get_instance()

        while not instance.event_queue.empty() and instance.processed_events < number_of_events:
            event = instance.event_queue.get(block=False)
            if event.type is not EventType.CANCELLED:
                instance.current_time = event.time_at
                logger.debug(
                    "%4.5f Event[id=%s, type=%s, time=%4.5f] executed; queue_size=%s",
                    instance.get_current_time(),
                    event.event_id,
                    event.type.name,
                    event.time_at,
                    instance.event_queue.qsize(),
                )
                event.action()
                instance.processed_events += 1

        if instance.event_queue.empty():
            logger.warning("%4.5f Simulation ended: empty event queue", instance.get_current_time())
        else:
            logger.info(
                "%4.5f Simulation ended: processed %s events",
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
        self.event_queue: PriorityQueue[Event] = PriorityQueue()
        self.event_lookup: dict[int, Event] = {}

        self.processed_events: int = 0

    def _get_next_id(self):
        """Return the next unique event identifier."""
        new_id = self.event_id_counter
        self.event_id_counter += 1
        return new_id


class Request:
    """Class for requests in the network."""

    next_req_id = 0

    def __init__(
        self,
        flow_id: int,
        arrival_time: float,
        total_size: float,
        min_resources: int,
    ):
        """
        Initialize a request.

        :param flow_id: Unique identifier of the flow that created the request.
        :param arrival_time: Time at which the request is generated.
        :param total_size: Total size of the request in resource units * time.
        :param min_resources: Minimum resources required to serve the request.
        """
        # Unique identifier of the request
        self.request_id: int = self.__class__.next_req_id
        self.__class__.next_req_id += 1

        self.flow_id: int = flow_id
        self.arrival_time: float = arrival_time
        self.total_size: float = total_size
        self.min_resources: int = min_resources

        # Identifier for the service event
        self.service_event_id: Optional[int] = None
        # Size of the request that has been served so far
        self.served_size: float = 0
        # Time of the last resource allocation
        self.last_alloc_time: float = self.arrival_time
        # Current resource allocation for the request
        self.current_alloc_resource: int = 0
        # Time at which the request service is completed
        self.end_service_time: float = 0.0

    def __str__(self) -> str:
        sep = "\n\t\t"
        req_str = (
            f"\n\tRequest[{sep}id={self.request_id}, flow_id={self.flow_id}, "
            f"event_id={self.service_event_id},"
            f"{sep}size={self.total_size:2.5f}, served_size={self.served_size:2.5f},"
            f"{sep}alloc_res={self.current_alloc_resource}\n\t]"
        )
        return req_str

    def _update_served_size_and_last_alloc_time(self):
        """Updates the served size and last allocation time of the request.
        Does not change the current allocation resources.
        """
        sim_time = Simulator.get_current_time()
        served_size = (sim_time - self.last_alloc_time) * self.current_alloc_resource

        logger.debug(
            "%4.5f %s: updated:  new served_size=%2.5f",
            sim_time,
            str(self),
            self.served_size + served_size,
        )

        self.served_size += served_size
        self.last_alloc_time = sim_time

    def realloc_resources(self, new_alloc_resources: int) -> float:
        """Reallocate resources for the request and return the remaining service time.

        :param new_alloc_resources: New resource allocation for the request
        :return: Remaining service time for the request
        """
        self._update_served_size_and_last_alloc_time()

        logger.debug(
            "%4.5f %s: update: new alloc_res = %s",
            Simulator.get_current_time(),
            str(self),
            new_alloc_resources,
        )

        self.current_alloc_resource = new_alloc_resources
        service_time_left = (self.total_size - self.served_size) / self.current_alloc_resource

        if service_time_left < 0:
            logger.error(
                "%4.5f %s: negative service time left: %4.5f",
                Simulator.get_current_time(),
                str(self),
                service_time_left,
            )

        return service_time_left

    def finish_service(self):
        """Finish servicing request"""
        self._update_served_size_and_last_alloc_time()
        self.current_alloc_resource = 0

        logger.debug("%4.5f %s: finish service", Simulator.get_current_time(), str(self))


class Flow(StatsBaseClass, ABC):
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
        super().__init__()

        self.flow_id = flow_id
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.on_arrival_callback = on_arrival_callback
        self.is_active = False

        # some statistics
        self.stats_start_time = 0
        self.total_gen_requests = 0
        self.total_accepted_requests = 0
        self.total_serviced_requests = 0
        self.total_rejected_requests = 0

        self.request_service_time: float = 0
        self.request_mean_resources: float = 0

    def set_on_arrival_callback(self, fn: Callable[[Request], None]):
        """Set the callback function to be executed when a request arrives."""
        self.on_arrival_callback = fn

    def generate_next_arrival_time(self) -> float:
        """Generates the time until the next request using exponential distribution."""
        return random.expovariate(self.arrival_rate)  # 1 / (1 / arrival_intensity)

    def _generate_request(self):
        """Generates a single request, updates stats, calls callback."""
        size = self._generate_service_time() * self._get_min_resources()
        req = Request(
            flow_id=self.flow_id,
            arrival_time=Simulator.get_current_time(),
            total_size=size,
            min_resources=self._get_min_resources(),
        )

        logger.debug("%4.5f %s: generated", Simulator.get_current_time(), str(req))

        self.request_generated()

        if self.on_arrival_callback is not None:
            self.on_arrival_callback(req)
        else:
            assert False, "callback not set"
            # add log that on_arrival_callback not set

    def _schedule_request(self):
        """Schedules the request arrival event."""
        arrival_time = self.generate_next_arrival_time() + Simulator.get_current_time()
        Simulator.schedule_event(
            time_at=arrival_time,
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

    def _start_stats_collection(self):
        """Starts the statistics collection."""
        self.stats_start_time = Simulator.get_current_time()

    def get_stats(self):
        """Return the statistics collected for the flow."""
        return {
            "stats_start_time": self.stats_start_time,
            "total_gen_requests": self.total_gen_requests,
            "total_accepted_requests": self.total_accepted_requests,
            "total_serviced_requests": self.total_serviced_requests,
            "total_rejected_requests": self.total_rejected_requests,
            "request_service_time": self.request_service_time,
            "request_mean_resources": self.request_mean_resources,
        }

    @StatsBaseClass.if_stats_enabled
    def request_accepted(self):
        """Increments the number of accepted requests."""
        self.total_accepted_requests += 1

    @StatsBaseClass.if_stats_enabled
    def request_rejected(self):
        """Increments the number of rejected requests."""
        self.total_rejected_requests += 1

    @StatsBaseClass.if_stats_enabled
    def request_serviced(self, req: Request):
        """Increments the number of serviced requests and stores the QoS stats."""
        service_time = req.end_service_time - req.arrival_time

        # recursive mean:
        total = self.total_serviced_requests
        self.request_service_time = (total * self.request_service_time + service_time) / (total + 1)

        self.request_mean_resources = (
            total * self.request_mean_resources + req.total_size / service_time
        ) / (total + 1)

        self.total_serviced_requests += 1

    @StatsBaseClass.if_stats_enabled
    def request_generated(self):
        """Increment the number of generated requests."""
        self.total_gen_requests += 1

    @abstractmethod
    def generate_request(self):
        """Generates a new request and schedules the next request arrival."""
        raise NotImplementedError

    @abstractmethod
    def _get_min_resources(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _generate_service_time(self) -> float:
        raise NotImplementedError


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

    def generate_request(self):
        """Generates a new request and schedules the next request arrival."""
        self._generate_request()
        self._schedule_request()


class ElasticDataFlow(Flow):
    """Class for elastic data flows in the network."""

    def __init__(
        self,
        flow_id: int,
        arrival_rate: float,
        service_rate: float,
        min_max_resources_range: tuple[int, int],
        requests_batch_probs: list[float],
    ):
        super().__init__(flow_id, arrival_rate, service_rate)
        self.min_resources, self.max_resources = min_max_resources_range
        self.requests_batch_probs = requests_batch_probs

        # specific to elastic data stats
        self.total_batches = 0
        self.requests_per_batch = 0

    def get_max_resources(self) -> int:
        """Return maximum amount of resources to serve the request"""
        return self.max_resources

    def _get_min_resources(self) -> int:
        """Return minimum resources required to serve the request."""
        return self.min_resources

    def _generate_service_time(self) -> float:
        """Generates the service time using exponential distribution."""
        return random.expovariate(self.service_rate)

    def generate_request(self):
        """Generate a batch of requests."""
        batch_size = random.choices(
            population=np.arange(1, len(self.requests_batch_probs) + 1, dtype=int),
            weights=self.requests_batch_probs,
        )[0]

        logger.debug(
            "%4.5f Flow[id=%s]: generate a batch of requests %d",
            Simulator.get_current_time(),
            self.flow_id,
            batch_size,
        )

        self.batch_generated(batch_size)

        for _ in range(batch_size):
            self._generate_request()
        self._schedule_request()

    @StatsBaseClass.if_stats_enabled
    def batch_generated(self, batch_size):
        """Increments the number of generated batches."""
        total = self.total_batches
        self.requests_per_batch = (total * self.requests_per_batch + batch_size) / (total + 1)
        self.total_batches += 1

    def get_stats(self):
        general_stats = super().get_stats()
        general_stats["mean_requests_per_batch"] = self.requests_per_batch

        return general_stats


class Network(StatsBaseClass):
    """Class for the network model."""

    def __init__(self, params: ParametersSet):
        """Initialize network parameters.
        :param params: Network parameters
        """
        super().__init__()
        self.params: ParametersSet = params
        self.capacity = self.params.beam_capacity

        self.requests_lookup: dict[int, Request] = {}
        self.flows_lookup: dict[int, Flow] = {}

        # Statistics

        # statistics collection start time
        self.stats_start_time = 0
        # last time statistics were updated
        self.last_stats_update = 0
        # allocated_resources * time
        self.mean_utilization = 0
        # {flow_id: num_reqs * time}
        self.mean_flow_reqs_in_service: dict[int, float] = {}
        # {flow_id: num_resources * time}
        self.mean_flow_res_in_service: dict[int, float] = {}

    def _start_stats_collection(self):
        """Start the statistics collection."""
        self.stats_start_time = Simulator.get_current_time()
        self.last_stats_update = Simulator.get_current_time()

        self.mean_utilization = 0
        self.mean_flow_reqs_in_service = {flow_id: 0 for flow_id in self.flows_lookup}
        self.mean_flow_res_in_service = {flow_id: 0 for flow_id in self.flows_lookup}

    def get_stats(self):
        """Return the statistics collected for the network."""

        # average over simulation time
        duration = Simulator.get_current_time() - self.stats_start_time
        mean_utilization = self.mean_utilization / duration
        mean_flow_reqs_in_service = {
            flow_id: val / duration for flow_id, val in self.mean_flow_reqs_in_service.items()
        }
        mean_flow_res_in_service = {
            flow_id: val / duration for flow_id, val in self.mean_flow_res_in_service.items()
        }

        return {
            "stats_start_time": self.stats_start_time,
            "last_stats_update": self.last_stats_update,
            "mean_utilization": mean_utilization,
            "mean_flow_reqs_in_service": mean_flow_reqs_in_service,
            "mean_flow_res_in_service": mean_flow_res_in_service,
        }

    @StatsBaseClass.if_stats_enabled
    def update_utilization_stats(self):
        """Update the utilization statistics."""
        current_time = Simulator.get_current_time()
        if current_time == self.last_stats_update:
            return

        duration = current_time - self.last_stats_update
        utilization = 0
        flow_reqs_in_service = {flow_id: 0 for flow_id in self.flows_lookup}
        flow_res_in_service = {flow_id: 0 for flow_id in self.flows_lookup}

        for req in self.requests_lookup.values():
            flow_id = req.flow_id
            flow_reqs_in_service[flow_id] += 1
            flow_res_in_service[flow_id] += req.current_alloc_resource
            utilization += req.current_alloc_resource

        for flow_id in self.flows_lookup:
            self.mean_flow_reqs_in_service[flow_id] += duration * flow_reqs_in_service[flow_id]
            self.mean_flow_res_in_service[flow_id] += duration * flow_res_in_service[flow_id]

        self.mean_utilization += duration * utilization
        self.last_stats_update = current_time

    def add_flows(self, n_real_time_flows: int, n_elastic_flows: int):
        """Add real-time data and elastic data flows to the network.
        Only one elastic data flow is supported.
        """
        if n_real_time_flows < 0 or n_elastic_flows != 1:
            logger.error(
                "Incorrect number of flows: rt flows = %s, data flows = %s",
                n_real_time_flows,
                n_elastic_flows,
            )
        flow_id = 0

        for i in range(n_real_time_flows):
            self.flows_lookup[flow_id] = RealTimeFlow(
                flow_id,
                self.params.real_time_lambdas[i],
                self.params.real_time_mus[i],
                self.params.real_time_resources[i],
            )
            flow_id += 1

        for _ in range(n_elastic_flows):
            self.flows_lookup[flow_id] = ElasticDataFlow(
                flow_id,
                self.params.data_lambda,
                self.params.data_mu,
                (self.params.data_resources_min, self.params.data_resources_max),
                self.params.data_requests_batch_probs,
            )
            flow_id += 1

        assert n_real_time_flows + n_elastic_flows == len(self.flows_lookup)
        logger.info("Created flows: RT = %s, Data = %s", n_real_time_flows, n_elastic_flows)

        for flow in self.flows_lookup.values():
            flow.set_on_arrival_callback(self.request_arrival_handler)
            flow.start_flow()

    def request_arrival_handler(self, new_req: Request):
        """Handle the arrival of a new request: acept or reject."""
        # required resources for existing requests
        min_required_resources = sum(req.min_resources for req in self.requests_lookup.values())
        its_flow = self.flows_lookup[new_req.flow_id]

        req_str = f"{Simulator.get_current_time():4.5f} " + str(new_req)

        if new_req.min_resources + min_required_resources <= self.capacity:
            logger.debug("%s: %s", req_str, "accepted")
            self.accept_request(new_req)
            its_flow.request_accepted()
        else:
            logger.debug("%s: %s", req_str, "rejected")
            its_flow.request_rejected()

    def accept_request(self, new_req: Request):
        """Process accepting a new request."""
        self.update_utilization_stats()
        its_flow = self.flows_lookup[new_req.flow_id]
        self.requests_lookup[new_req.request_id] = new_req

        if isinstance(its_flow, RealTimeFlow):
            service_time = new_req.realloc_resources(its_flow.fixed_resource)
            # schedule service event
            event_id = Simulator.schedule_event(
                time_at=Simulator.get_current_time() + service_time,
                action=partial(self.request_service_handler, new_req.request_id),
                e_type=EventType.REQUEST_SERVICE,
            )
            new_req.service_event_id = event_id

        self.redistribute_elastic_resources()

    def _log_str_with_distr(self, mark: str = ""):
        """Return a string representation of resource distibution in the network."""
        str_start = f"{Simulator.get_current_time():4.5f} Distribution [{mark}]:\n"

        for _, req in self.requests_lookup.items():
            req_str = (
                f"\tRequest[id={req.request_id}, flow_id={req.flow_id}, "
                f"event_id={req.service_event_id}, alloc_res={req.current_alloc_resource}]\n"
            )
            str_start += req_str

        return str_start

    def redistribute_elastic_resources(self):
        """Adjust resource allocation among elastic requests."""
        self.update_utilization_stats()

        min_required_resources = sum(req.min_resources for req in self.requests_lookup.values())
        if min_required_resources > self.capacity:
            logger.error(
                "Redistribution failed: total required resources exceed capacity: %s > %s",
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

        logger.debug(
            "%4.5f Resource distribution: min_required=%s, rt_reqs=%s, ed_reqs=%s",
            Simulator.get_current_time(),
            min_required_resources,
            len(real_time_requests),
            len(elastic_data_requests),
        )

        logger.debug("%s", self._log_str_with_distr("old"))

        if len(elastic_data_requests) == 0:
            return

        real_time_resources = sum(req.current_alloc_resource for req in real_time_requests)
        resources_left_for_elastic = self.capacity - real_time_resources

        # resources per request rounded down: q
        rfloor_res = resources_left_for_elastic // len(elastic_data_requests)
        # number of requests with (q + 1) resources
        num_req_ceil_res = resources_left_for_elastic % len(elastic_data_requests)
        # number of requests with (q) resources
        num_req_floor_res = len(elastic_data_requests) - num_req_ceil_res

        total_distributed = rfloor_res * num_req_floor_res + (rfloor_res + 1) * num_req_ceil_res

        if resources_left_for_elastic != total_distributed:
            logger.error("%4.5f Incorrect distribution", Simulator.get_current_time())

        # TBD: check min according to flow
        for req in elastic_data_requests:
            its_flow = self.flows_lookup[req.flow_id]
            assert isinstance(its_flow, ElasticDataFlow)

            suggested_resources = rfloor_res + 1 if num_req_ceil_res > 0 else rfloor_res
            num_req_ceil_res -= 1
            alloc_resources = min(suggested_resources, its_flow.get_max_resources())
            # allocate new resources to request if needed
            if req.current_alloc_resource != alloc_resources:
                new_service_time = req.realloc_resources(alloc_resources)
                # cancel previous service event
                if req.service_event_id is not None:
                    Simulator.cancel_event(req.service_event_id)
                # schedule new service event
                event_id = Simulator.schedule_event(
                    time_at=Simulator.get_current_time() + new_service_time,
                    action=partial(self.request_service_handler, req.request_id),
                    e_type=EventType.REQUEST_SERVICE,
                )
                req.service_event_id = event_id

        logger.debug(self._log_str_with_distr("new"))

    def request_service_handler(self, req_id: int):
        """Handle the service completion of request."""
        self.update_utilization_stats()
        req = self.requests_lookup[req_id]
        req.end_service_time = Simulator.get_current_time()
        req.finish_service()

        if not np.isclose(req.served_size, req.total_size):
            logger.error(
                "%4.5f Request[id=%s]: served(%2.5f) != total(%2.5f)",
                Simulator.get_current_time(),
                req.request_id,
                req.served_size,
                req.total_size,
            )

        self.flows_lookup[req.flow_id].request_serviced(req)

        # delete request
        del self.requests_lookup[req_id]
        # update resource distribution
        self.redistribute_elastic_resources()


def convert_stats_to_metrics(params: ParametersSet, network: Network):
    """Convert network statistics to metrics."""
    net_stats = network.get_stats()
    n_rt_flows = params.real_time_flows

    flows_stats = {
        flow_id: network.flows_lookup[flow_id].get_stats() for flow_id in network.flows_lookup
    }

    has_ed = any(isinstance(flow, ElasticDataFlow) for flow in network.flows_lookup.values())
    has_rt = any(isinstance(flow, RealTimeFlow) for flow in network.flows_lookup.values())

    metrics = Metrics(
        rt_request_rej_prob=[0] * n_rt_flows,
        mean_rt_requests_in_service=[0] * n_rt_flows,
        mean_resources_per_rt_flow=[0] * n_rt_flows,
    )

    if has_rt:
        for i in range(n_rt_flows):
            stats = flows_stats[i]

            metrics.rt_request_rej_prob[i] = (
                stats["total_rejected_requests"] / stats["total_gen_requests"]
            )

            metrics.mean_rt_requests_in_service[i] = net_stats["mean_flow_reqs_in_service"][i]

            metrics.mean_resources_per_rt_flow[i] = net_stats["mean_flow_res_in_service"][i]

    if has_ed:
        key = n_rt_flows if has_rt else 0
        ed_stats = flows_stats[key]
        metrics.data_request_rej_prob = (
            ed_stats["total_rejected_requests"] / ed_stats["total_gen_requests"]
        )

        metrics.mean_data_request_service_time = ed_stats["request_service_time"]
        metrics.mean_resources_per_data_request = ed_stats["request_mean_resources"]
        metrics.mean_data_requests_in_service = net_stats["mean_flow_reqs_in_service"][key]
        metrics.mean_resources_per_data_flow = net_stats["mean_flow_res_in_service"][key]
        metrics.mean_data_requests_per_batch = ed_stats["mean_requests_per_batch"]

    metrics.beam_utilization = net_stats["mean_utilization"]

    return metrics


def get_argparser():
    """Return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Satellite network simulation: n real-time flows, 1 elastic data flow"
    )

    # Real-time flow parameters
    parser.add_argument(
        "--real_time_flows",
        type=int,
        default=2,
        help="Number of real-time data flows in the network",
    )
    parser.add_argument(
        "--real_time_lambdas",
        type=float,
        nargs="+",
        default=[15.0, 3.0],
        help="Arrival rates (lambda) for real-time data flows (space-separated list)",
    )
    parser.add_argument(
        "--real_time_mus",
        type=float,
        nargs="+",
        default=[0.2, 0.2],
        help="Service rates (mu) for real-time data flows (space-separated list)",
    )
    parser.add_argument(
        "--real_time_resources",
        type=int,
        nargs="+",
        default=[1, 5],
        help="Resource units required by each real-time flow (space-separated list)",
    )

    # Elastic data flow parameters
    parser.add_argument(
        "--data_resources_min",
        type=int,
        default=1,
        help="Minimum resource units allocated for elastic data flow",
    )
    parser.add_argument(
        "--data_resources_max",
        type=int,
        default=10,
        help="Maximum resource units allocated for elastic data flow",
    )
    parser.add_argument(
        "--data_lambda",
        type=float,
        default=2.5,
        help="Arrival rate (lambda) for elastic data flow",
    )
    parser.add_argument(
        "--data_mu",
        type=float,
        default=1.0,
        help="Service rate (mu) for elastic data flow",
    )
    parser.add_argument(
        "--data_requests_batch_probs",
        type=float,
        nargs="+",
        default=[1 / 3, 1 / 3, 1 / 3],
        help="Batch probs f_s, s = 1, ..., B",
    )
    # Beam capacity parameter
    parser.add_argument(
        "--beam_capacity",
        type=int,
        default=100,
        help="Total beam capacity in resource units",
    )
    # Has elastic data flow
    parser.add_argument(
        "--elastic_data_flow",
        action="store_true",
        default=False,
        help="Flag to include elastic data flow in the network",
    )

    # Simulation parameters
    parser.add_argument(
        "--warmup",
        type=int,
        default=1000,
        help="Number of events to run before starting statistics collection",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=10000,
        help="Total number of events to simulate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulation reproducibility",
    )

    return parser


def main():
    """ Main function. Parses args from command line and runs simulation"""
    parser = get_argparser()
    args = parser.parse_args()

    random.seed(args.seed)
    params = ParametersSet(
        real_time_flows=args.real_time_flows,
        real_time_lambdas=args.real_time_lambdas,
        real_time_mus=args.real_time_mus,
        real_time_resources=args.real_time_resources,
        data_resources_min=args.data_resources_min,
        data_resources_max=args.data_resources_max,
        data_lambda=args.data_lambda,
        data_mu=args.data_mu,
        data_requests_batch_probs=args.data_requests_batch_probs,
        beam_capacity=args.beam_capacity,
    )

    logger.info("%s", params)

    simulator = Simulator.get_instance()
    network = Network(params)
    network.add_flows(
        n_real_time_flows=params.real_time_flows, n_elastic_flows=int(args.elastic_data_flow)
    )

    warmup_start_time = time.time()
    simulator.run(args.warmup)
    warmup_finish_time = time.time()

    logger.info("Warmup time: %4.5f", warmup_finish_time - warmup_start_time)

    # start statistics collection
    network.enable_stats_collection()
    for flow in network.flows_lookup.values():
        flow.enable_stats_collection()

    time_start = time.time()
    simulator.run(args.warmup + args.events)
    time_finish = time.time()

    logger.info("Simulation time: %4.5f", time_finish - time_start)

    metrics = convert_stats_to_metrics(params, network)
    logger.info("%s", metrics)

    print(metrics)


if __name__ == "__main__":
    main()
